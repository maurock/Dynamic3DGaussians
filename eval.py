"""Class to evaluate a single output on the test set."""
import os
import argparse
from utils import utils_metrics, utils_mesh
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

        # Extract ground truth data for evaluation
        # self.rgb_gt = self.get_rgb_gt()
        # self.depth_gt = self.get_depth_gt()
        # self.pc_gt = self.get_pc_gt()

        # Extract prediction data for evaluation
        if not self.check_pred_exists():
            rgb_pred_npy, depth_pred_npy, pc_pred_npy = extract_output_data.extract_output_data(
                self.input_seq, self.args.exp_name, self.args.output_seq
            )                         
            extract_output_data.save_output_pointcloud(pc_pred_npy, self.args.exp_name, self.args.output_seq)
            extract_output_data.save_output_depth_images(depth_pred_npy, self.args.exp_name, self.args.output_seq)
            extract_output_data.save_output_rgb_images(rgb_pred_npy, self.args.exp_name, self.args.output_seq)
        # else:
        #     rgb_pred_npy, depth_pred_npy, pc_pred_npy = self.load_predictions()
        # # self.rgb_pred = torch.tensor(rgb_pred_npy).cuda()
        # self.depth_pred = torch.tensor(depth_pred_npy).cuda()
        # self.pc_pred = torch.tensor(pc_pred_npy).cuda()

        # Normalise depths
        # self.depth_pred = helpers.normalise_depth(self.depth_pred, min_depth=1, max_depth=6)
        # self.depth_gt = helpers.normalise_depth(self.depth_gt, min_depth=1, max_depth=6)
        print()


    def get_data_dir(self):
        """Get the data directory from eval_helper.txt stored during training"""
        eval_helper_path = os.path.join(self.experiment_dir, "eval", "eval_helper.txt")
        with open(eval_helper_path, "r") as f:
            input_seq = f.readline().strip()        
        return input_seq, os.path.join(os.path.dirname(data.__file__), input_seq)    
    
    def get_rgb_gt(self):
        """Get the ground truth rgb images.
        Return:
            rgb_images: torch tensor of shape (num_images, H, W, 3)
        """
        meta_path = os.path.join(self.data_dir, "test_meta.json")
        with open(meta_path, 'r') as file:
            meta = json.load(file)
        rgb_paths = [os.path.join(self.data_dir, 'ims', x) for x in meta['fn'][0]]
        # Convert image path to torch tensor
        rgb_images = torch.stack([helpers.load_rgb_image(x).permute(1, 2, 0) for x in rgb_paths],dim=0)
        return rgb_images
    
    def get_depth_gt(self):
        """Get the ground truth depth images.
        Return:
            rgb_images: torch tensor of shape (num_images, 3, H, W)
        """
        meta_path = os.path.join(self.data_dir, "test_meta.json")
        with open(meta_path, 'r') as file:
            meta = json.load(file)
        disp_paths = [os.path.join(self.data_dir, 'depth', x.split(os.sep)[0], 'depth.tiff') for x in meta['fn'][0]]
        # self._debug_depth(depth_paths)
        # Convert image path to torch tensor
        depth_images = torch.stack([helpers.load_disparity_image(x) for x in disp_paths], dim=0)
        depth_images = helpers.convert_disparity_to_depth(depth_images)
        return depth_images
    
    def _debug_depth(self, depth_paths):
        image = Image.open(depth_paths[0])
        image_array = np.array(image)
        image_array = image_array.astype(np.float32) * 255
        image = Image.fromarray(image_array)
        image.show()

    def get_pc_gt(self):
        """Get the ground truth point cloud."""
        input_seq = self.data_dir.split(os.sep)[-1]
        mesh_gt = trimesh.load_mesh(
            os.path.join(
                os.path.dirname(data.__file__),
                'refnerf-blend', 'obj', f'{input_seq}.obj'
            )
        )
        mesh_gt = utils_mesh.as_mesh(mesh_gt)
        pc_gt = trimesh.sample.sample_surface(mesh_gt, 50000)[0]
        pc_gt = torch.tensor(pc_gt).cuda()
        return pc_gt
    
    def check_pred_exists(self):
        """Check if the prediction data exists."""
        pred_exists = (os.path.exists(os.path.join(self.experiment_dir, "eval", "rgb_pred.npz")) and
                  os.path.exists(os.path.join(self.experiment_dir, "eval", "depth_pred.npz")) and
                  os.path.exists(os.path.join(self.experiment_dir, "eval", "pc_pred.npz")))
        return pred_exists

    def load_rgb_pred(self):
        """Load the prediction data."""
        # Load data as numpy arrays
        rgb_pred_npy = np.load(os.path.join(self.experiment_dir, "eval", "rgb_pred.npz"))['rgb']
        return rgb_pred_npy
    
    def load_depth_pred(self):
        """Load the prediction data."""
        # Load data as numpy arrays
        depth_pred_npy = np.load(os.path.join(self.experiment_dir, "eval", "depth_pred.npz"))['depth']
        return depth_pred_npy
    
    def load_pc_pred(self):
        """Load the prediction data."""
        # Load data as numpy arrays
        pc_pred_npy = np.load(os.path.join(self.experiment_dir, "eval", "pc_pred.npz"))['pts']
        return pc_pred_npy
    
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

    def evaluate_rgb(self):
        rgb_gt = self.get_rgb_gt()
        rgb_pred_npy = self.load_rgb_pred()
        rgb_pred = torch.tensor(rgb_pred_npy).cuda()
        ssim_rgb = external.calc_ssim(rgb_gt, rgb_pred.clip(0,1)).item()
        psnr_rgb = external.calc_psnr(rgb_gt, rgb_pred.clip(0,1)).mean().item()
        torch.cuda.empty_cache()

        return ssim_rgb, psnr_rgb
    
    def evaluate_3D(self):
        pc_gt = self.get_pc_gt()
        pc_pred_npy = self.load_pc_pred()
        pc_pred = torch.tensor(pc_pred_npy).cuda()

        cd = chamfer_distance(pc_gt.unsqueeze(0), pc_pred.unsqueeze(0))[0].item()
        emd = earth_mover_distance(pc_gt[:5000].cpu(), pc_pred[:5000].cpu()).item()
        torch.cuda.empty_cache()

        return cd, emd
    
    def evaluate_depth(self):
        # Load
        depth_gt = self.get_depth_gt()
        depth_pred_npy = self.load_depth_pred()
        depth_pred = torch.tensor(depth_pred_npy).cuda()
        
        # Normalise depth
        depth_pred = helpers.normalise_depth(depth_pred, min_depth=1, max_depth=6)
        depth_gt = helpers.normalise_depth(depth_gt, min_depth=1, max_depth=6)
        ssim_depth = external.calc_ssim(depth_gt, depth_pred).item()
        psnr_depth = external.calc_psnr(depth_gt, depth_pred).mean().item()
       
        torch.cuda.empty_cache()
        return ssim_depth, psnr_depth
    
def main(args):
    evaluator = Evaluator(args)

    # evaluator._debug_shapes()

    # Compute RGB metrics
    ssim_rgb, psnr_rgb = evaluator.evaluate_rgb()
    # TODO LPIPS

    # Compute 3D metrics
    cd, emd = evaluator.evaluate_3D()

    # Compute depth metrics
    ssim_depth, psnr_depth = evaluator.evaluate_depth()

    # Save metrics as json file
    metrics = {
        'ssim_rgb': ssim_rgb,
        'psnr_rgb': psnr_rgb,
        'cd': cd,
        'emd': emd,
        'ssim_depth': ssim_depth,
        'psnr_depth': psnr_depth
    }
    with open(os.path.join(evaluator.experiment_dir, "eval", "metrics.json"), 'w') as file:
        json.dump(metrics, file, indent=4)

    # Print metrics
    for k, v in metrics.items():
        print(f'{k}: {v}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="RefNeRF", help="Dataset name. Choose between 'RefNeRF'")
    parser.add_argument("--exp_name", type=str, default="", help="Path to the experiment directory inside output/, e.g. exp1")
    parser.add_argument("--output_seq", type=str, default="", help="Path to the run directory inside output/<exp>, e.g. toaster")
    args = parser.parse_args()

    args.exp_name = 'exp1'
    args.output_seq = 'toaster_15000_new_smooth01'

    main(args)
