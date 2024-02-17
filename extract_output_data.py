
import torch
import numpy as np
import time
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from helpers import setup_camera
import os
import data
import json
import matplotlib.cm as cm
import output
import argparse
import helpers
from utils import utils_data
"""
Visualiser. This method was modified from the original code for better interactivity.
- Press 'C' to change camera
- Press 'M' to change mode (color or depth)
"""

torch.cuda.empty_cache()

RENDER_MODE = 'color'  # 'color', 'depth' or 'centers'
# RENDER_MODE = 'color'  # 'color', 'depth' or 'centers'
# RENDER_MODE = 'centers'  # 'color', 'depth' or 'centers'

ADDITIONAL_LINES = None  # None, 'trajectories' or 'rotations'
# ADDITIONAL_LINES = 'trajectories'  # None, 'trajectories' or 'rotations'
# ADDITIONAL_LINES = 'rotations'  # None, 'trajectories' or 'rotations'

# REMOVE_BACKGROUND = False  # False or True
REMOVE_BACKGROUND = False  # False or True

FORCE_LOOP = False  # False or True
# FORCE_LOOP = True  # False or True

w, h = 800, 800 #640*2, 360*2 # Needs to be the same as W, H of the training images
near, far = 0.00001, 10000.0
view_scale = 1 # 3.9
fps = 20
traj_frac = 25  # 4% of points
traj_length = 15
def_pix = torch.tensor(
    np.stack(np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5, 1), -1).reshape(-1, 3)).cuda().float()
pix_ones = torch.ones(h * w, 1).cuda().float()
colormappa = cm.magma #cm.BuPu_r  # You can change this to cm.jet, cm.viridis, etc.


def load_scene_data(seq, exp, seg_as_col=False):
    params = dict(np.load(f"{os.path.dirname(output.__file__)}/{exp}/{seq}/params.npz"))
    params = {k: torch.tensor(v).cuda().float() for k, v in params.items()}
    is_fg = params['seg_colors'][:, 0] > 0.5
    scene_data = []
    for t in range(len(params['means3D'])):
        rendervar = {
            'means3D': params['means3D'][t],
            # 'colors_precomp': params['rgb_colors'][t] if not seg_as_col else params['seg_colors'],
            'shs': params['shs'],
            'rotations': torch.nn.functional.normalize(params['unnorm_rotations'][t]),
            'opacities': torch.sigmoid(params['logit_opacities']),
            'scales': torch.exp(params['log_scales']),
            'means2D': torch.zeros_like(params['means3D'][0], device="cuda")
        }
        if REMOVE_BACKGROUND:
            rendervar = {k: v[is_fg] for k, v in rendervar.items()}
        scene_data.append(rendervar)
    if REMOVE_BACKGROUND:
        is_fg = is_fg[is_fg]
    return scene_data, is_fg


def rgbd2pcd(im, depth, w2c, k, show_depth=False, project_to_cam_w_scale=None):
    invk = torch.inverse(torch.tensor(k).cuda().float())
    c2w = torch.inverse(torch.tensor(w2c).cuda().float())
    radial_depth = depth[0].reshape(-1)
    # def_rays is the unnormalised rays in the camera frame
    # x_2D = K @ x_3D, so x_3D = K^-1 @ x_2D. In this case, x_2D is pixels and x_3D is rays.
    def_rays = (invk @ def_pix.T).T
    # I think we should use the unnnormalised version of it
    def_radial_rays = def_rays
    # def_radial_rays = def_rays / torch.linalg.norm(def_rays, ord=2, dim=-1)[:, None]
    # rays * depth = 3D points in camera coords
    pts_cam = def_radial_rays * radial_depth[:, None]
    z_depth = pts_cam[:, 2]
    if project_to_cam_w_scale is not None:
        pts_cam = project_to_cam_w_scale * pts_cam / z_depth[:, None]
    pts4 = torch.concat((pts_cam, pix_ones), 1)
    # pts is points in 3D world coords
    pts = (c2w @ pts4.T).T[:, :3]
    
    return pts


def get_camera_positions(seq):
    path = os.path.join(os.path.dirname(data.__file__), seq, 'test_meta.json')
    with open(path, 'r') as file:
        cameras = json.load(file)
    camera_positions = []
    for i in range(np.array(cameras["k"]).shape[1]):
        k = np.array(cameras["k"])[0, i, :, :]
        w2c = np.array(cameras["w2c"])[0, i, :, :]
        camera_positions.append([w2c, k])
    
    return camera_positions


def save_output_pointcloud(pts_all, exp_name, output_seq):
    output_pc_path = f"{os.path.dirname(output.__file__)}/{exp_name}/{output_seq}/eval/pc_pred"
    np.savez_compressed(output_pc_path, pts=pts_all)


def save_output_depth_images(depth_all, exp_name, output_seq):
    output_depth_path = f"{os.path.dirname(output.__file__)}/{exp_name}/{output_seq}/eval/depth_pred"
    np.savez_compressed(output_depth_path, depth=depth_all)


def save_output_rgb_images(im_all, exp_name, output_seq):
    output_im_path = f"{os.path.dirname(output.__file__)}/{exp_name}/{output_seq}/eval/rgb_pred"
    np.savez_compressed(output_im_path, rgb=im_all)


def extract_output_data(input_seq, exp_name, output_seq, near=0.1, far=100000.0):
    """Extract pointcloud and depth images from the output of the model.
    Parameters:
        input_seq (str): Name of the input sequence, e.g. 'toaster'
        exp_name (str): Name of the experiment, e.g. 'exp1'
        output_seq (str): Name of the output sequence, e.g. 'toaster_15000'
    Returns:
        pts_all (np.array): Pointcloud of the scene, shape (N, 3)
        depth_all (np.array): Depth images of the scene, shape (M, H, W)
        im_all (np.array): RGB images of the scene, shape (M, H, W, 3)
    """
    scene_data, is_fg = load_scene_data(output_seq, exp_name)

    # With callback
    camera_index = [0]
    mode = [RENDER_MODE]
    camera_positions = get_camera_positions(input_seq)

    w2c, k = camera_positions[camera_index[0]]

    # im, depth = render(w2c, k, scene_data[0])
    im, depth, alpha = helpers.render(w, h, k, w2c, near, far, scene_data[0])
    init_pts = rgbd2pcd(im, depth, w2c, k, show_depth=(mode[0] == 'depth'))

    view_k = k * view_scale
    view_k[2, 2] = 1

    # Accumulate all points for plotting
    pts_all = []
    depth_all = []
    im_all = []

    # Set ratio_pointcloud (amount of total points to keep) based on the number of images in the test data
    num_images = len(camera_positions)
    num_required_points = 200000  # some of these be filtered as they are too far
    num_extracted_points = num_images * w * h
    ratio_pointcloud = num_required_points / num_extracted_points
    for i in range(len(camera_positions)):
       
        w2c, k = camera_positions[i]

        # im, depth = render(w2c, k, scene_data[0])
        im, depth, alpha = helpers.render(w, h, k, w2c, near, far, scene_data[0])
        pts = rgbd2pcd(im, depth, w2c, k, show_depth=(mode[0] == 'depth'))

        # Filter pointcloud: reduce the number of points
        pts_npy = np.array(pts.cpu())
        pts_npy = utils_data.filter_pointcloud(pts_npy, w2c, ratio_pointcloud)

        # Accumulate pointcloud
        pts_all.extend(pts_npy)

        # Accumulate depth images
        depth = depth.cpu().numpy()[0]
        depth_all.append(depth)

        # Accumulate rgb images
        im_all.append(im.cpu().numpy().transpose(1, 2, 0))

    im_all = np.array(im_all) 
    depth_all = np.array(depth_all)
    pts_all = np.array(pts_all)

    print(f"[INFO] Number of points extracted: {pts_all.shape[0]}")
    
    return im_all, depth_all, pts_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_depth", default=False, action='store_true', help="Extract predicted depth images"
    )
    parser.add_argument(
        "--save_pointcloud", default=False, action='store_true', help="Extract predicted pointcloud"
    )
    parser.add_argument(
        "--save_rgb", default=False, action='store_true', help="Extract predicted rgb images"
    )
    args = parser.parse_args()

    args.save_depth = True
    args.save_pointcloud = True
    args.save_rgb = True

    # Input
    input_seq = ''
    # Output
    exp_name = ""
    output_seq = ""

    # Visualise
    for sequence in [output_seq]: #, "boxes", "football", "juggle", "softball", "tennis"]:
        im_all, depth_all, pts_all = extract_output_data(input_seq, exp_name, sequence)

        # Save pointcloud
        if args.save_pointcloud:
            helpers.create_dirs(f"{os.path.dirname(output.__file__)}/{exp_name}/{output_seq}/eval")
            save_output_pointcloud(pts_all, exp_name, output_seq)

        # Save depth images
        if args.save_depth:
            helpers.create_dirs(f"{os.path.dirname(output.__file__)}/{exp_name}/{output_seq}/eval")
            save_output_depth_images(depth_all, exp_name, output_seq)

        # Save rgb images
        if args.save_rgb:
            helpers.create_dirs(f"{os.path.dirname(output.__file__)}/{exp_name}/{output_seq}/eval")
            save_output_rgb_images(im_all, exp_name, output_seq)
            
