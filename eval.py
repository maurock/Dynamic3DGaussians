"""Class to evaluate a single output on the test set."""
import os
import argparse
from utils import utils_mesh, utils_data
import data
import output
import json
import numpy as np
from PIL import Image
import torch
import helpers
import extract_output_data
import trimesh
import external
from pytorch3d.loss import chamfer_distance
import plotly.graph_objects as go
from utils.utils_metrics import earth_mover_distance

class Evaluator:
    def __init__(self, args) -> None:
        self.args = args
        self.dataset = args.dataset
        self.experiment_dir = os.path.join(os.path.dirname(output.__file__), self.args.exp_name, self.args.output_seq)
        self.input_seq, self.data_dir = self.get_data_dir()

        # Extract prediction data for evaluation
        if (not self.check_pred_exists()) and (self.args.save_eval_data):
            rgb_pred_npy, depth_pred_npy, pc_pred_npy = extract_output_data.extract_output_data(
                self.input_seq, self.args.exp_name, self.args.output_seq
            )                   
            if self.args.save_eval_data:      
                extract_output_data.save_output_pointcloud(pc_pred_npy, self.args.exp_name, self.args.output_seq)
                extract_output_data.save_output_depth_images(depth_pred_npy, self.args.exp_name, self.args.output_seq)
                extract_output_data.save_output_rgb_images(rgb_pred_npy, self.args.exp_name, self.args.output_seq)

    def get_data_dir(self):
        """Get the data directory from eval_helper.txt stored during training"""
        eval_helper_path = os.path.join(self.experiment_dir, "eval", "eval_helper.txt")
        with open(eval_helper_path, "r") as f:
            input_seq = f.readline().strip()        
        return input_seq, os.path.join(os.path.dirname(data.__file__), input_seq)    
       
    def _debug_depth(self, depth_paths):
        image = Image.open(depth_paths[0])
        image_array = np.array(image)
        image_array = image_array.astype(np.float32) * 255
        image = Image.fromarray(image_array)
        image.show()
    
    def check_pred_exists(self):
        """Check if the prediction data exists."""
        pred_exists = (os.path.exists(os.path.join(self.experiment_dir, "eval", "rgb_pred.npz")) and
                  os.path.exists(os.path.join(self.experiment_dir, "eval", "depth_pred.npz")) and
                  os.path.exists(os.path.join(self.experiment_dir, "eval", "pc_pred.npz")))
        return pred_exists  
    
    def _debug_shapes(self):
        print(f'rgb_gt: {self.rgb_gt.shape}')
        print(f'depth_gt: {self.depth_gt.shape}')
        print(f'pc_gt: {self.pc_gt.shape}')
        print(f'pc_pred: {self.pc_pred.shape}')
        print(f'depth_gt: {self.depth_gt.shape}')
        print(f'depth_pred: {self.depth_pred.shape}')

    def _debug_pc(self):
        fig = go.Figure(data=[go.Scatter3d(
            x=self.pc_gt[:, 0].cpu().numpy(),
            y=self.pc_gt[:, 1].cpu().numpy(),
            z=self.pc_gt[:, 2].cpu().numpy(),
            mode='markers',
            marker=dict(
                size=3,
                color='cyan',  # Different color for debug points
            ),
            name='GT Points'
        ),
            go.Scatter3d(
                x=self.pc_pred[:, 0].cpu().numpy(),
                y=self.pc_pred[:, 1].cpu().numpy(),
                z=self.pc_pred[:, 2].cpu().numpy(),
                mode='markers',
                marker=dict(
                    size=3,
                    color='red',  # Different color for debug points
                ),
                name='Pred Points'
            )])
        fig.show()

    def _debug_depth(self, depth_gt, depth_pred):
        for i in range(1):
            # Convert tensors to numpy arrays
            depth_gt_array = depth_gt[i].cpu().numpy()
            depth_pred_array = depth_pred[i].cpu().numpy()

            # Scale the arrays from 0-1 to 0-255 and convert to uint8
            depth_gt_array = (depth_gt_array * 255).astype(np.uint16)
            depth_pred_array = (depth_pred_array * 255).astype(np.uint16)

            # Create PIL images from the numpy arrays
            image_gt = Image.fromarray(depth_gt_array)
            image_pred = Image.fromarray(depth_pred_array)

            # Combine the two images side by side
            total_width = image_gt.width + image_pred.width

            # Create a new blank image with the correct combined size
            new_im = Image.new('L', (total_width, image_gt.height))

            # Paste the two images into the new image
            new_im.paste(image_gt, (0, 0))
            new_im.paste(image_pred, (image_gt.width, 0))

            # Display the combined image
            new_im.show()


    def evaluate_rgb(self, rgb_pred_npy):
        with torch.no_grad():
            # Load ground truth and predicted RGB images
            rgb_gt = utils_data.load_rgb_gt(self.data_dir, 'test')
            rgb_pred = torch.tensor(rgb_pred_npy).float().cuda()

            # Prepare for PSNR and SSIM
            rgb_gt = rgb_gt.permute(0, 3, 1, 2).float()
            rgb_pred = rgb_pred.permute(0, 3, 1, 2).float()

            # Compute metrics
            batch_size = 100
            ssims = []
            psnrs = []
            for i in range(0, len(rgb_gt), batch_size):
                ssims.append(external.calc_ssim(rgb_gt[i:i+batch_size], rgb_pred[i:i+batch_size]))
                psnrs.extend(external.calc_psnr(rgb_gt[i:i+batch_size], rgb_pred[i:i+batch_size]).mean(1))
            ssim_rgb = torch.stack(ssims).mean().item()
            psnr_rgb = torch.cat(psnrs).mean().item()
        torch.cuda.empty_cache()
        return ssim_rgb, psnr_rgb
    
    def evaluate_3D(self, pc_pred_npy):
        with torch.no_grad():
            # Load ground truth and predicted pointclouds
            pc_gt = utils_data.load_pointcloud_gt(self.data_dir)
            pc_pred = torch.tensor(pc_pred_npy).float().cuda()

            # Focus on bounding box around the object (to remove floaters)
            pc_pred = utils_data.get_points_in_bbox(pc_pred)

            # Compute metrics
            cd = chamfer_distance(pc_gt.unsqueeze(0), pc_pred.unsqueeze(0))[0].item()
            emd = earth_mover_distance(pc_gt[:5000].cpu(), pc_pred[:5000].cpu()).item()
        torch.cuda.empty_cache()

        return cd
    
    def evaluate_depth(self, depth_pred_npy):
        with torch.no_grad():
            if self.args.dataset == 'ShinyBlender':
                is_disparity = True
            elif self.args.dataset == 'GlossySynthetic':
                is_disparity = False
            else:
                raise ValueError('Invalid dataset name')
            # Load ground truth and predicted depth images
            depth_gt = utils_data.load_depth_gt(self.data_dir, 'test', is_disparity)
            depth_pred = torch.tensor(depth_pred_npy).float().cuda()
            
            # Normalise depth
            depth_pred = helpers.normalise_depth(depth_pred, min_depth=1, max_depth=6)
            depth_gt = helpers.normalise_depth(depth_gt, min_depth=1, max_depth=6)

            # Compute metrics
            ssim_depth = external.calc_ssim(depth_gt, depth_pred).item()
            psnr_depth = external.calc_psnr(depth_gt, depth_pred).mean().item()
       
        torch.cuda.empty_cache()
        return ssim_depth, psnr_depth
    

    def run_evaluation(self):
        torch.cuda.empty_cache()
        print('Evaluating...')

        if (not self.check_pred_exists()):
            rgb_pred_npy, depth_pred_npy, pc_pred_npy = extract_output_data.extract_output_data(
                self.input_seq, self.args.exp_name, self.args.output_seq
            )  
        else:
            rgb_pred_npy = utils_data.load_prediction(self.experiment_dir, 'rgb')
            depth_pred_npy = utils_data.load_prediction(self.experiment_dir, 'depth')
            pc_pred_npy = utils_data.load_prediction(self.experiment_dir, 'pointcloud')

        # Compute RGB metrics
        ssim_rgb, psnr_rgb = self.evaluate_rgb(rgb_pred_npy)
        # TODO LPIPS

        # Compute 3D metrics
        cd = self.evaluate_3D(pc_pred_npy)

        # Compute depth metrics
        # sim_depth, psnr_depth = self.evaluate_depth(depth_pred_npy)

        # Save metrics as json file
        metrics = {
            'ssim_rgb': ssim_rgb,
            'psnr_rgb': psnr_rgb,
            'cd': cd,
            #'emd': emd,
            #'ssim_depth': ssim_depth,
            #'psnr_depth': psnr_depth
        }
        with open(os.path.join(self.experiment_dir, "eval", "metrics.json"), 'w') as file:
            json.dump(metrics, file, indent=4)

        print("Evaluation complete. I hope it's what you expected!")
        # Print metrics
        for k, v in metrics.items():
            print(f'{k}: {v}')

def main(args):
    evaluator = Evaluator(args)
    evaluator.run_evaluation()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ShinyBlender", help="Dataset name. Choose between 'ShinyBlender', 'GlossySynthetic'")
    parser.add_argument("--exp_name", type=str, default="", help="Path to the experiment directory inside output/, e.g. exp1")
    parser.add_argument("--output_seq", type=str, default="", help="Path to the run directory inside output/<exp>, e.g. toaster")
    parser.add_argument("--save_eval_data", action="store_true", default=False, help="Save evaluation data")

    args = parser.parse_args()

    main(args)
