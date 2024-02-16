"""Script containing utility functions for data processing, data loading, and data creation."""
from PIL import Image as PIL_Image
import os
import data
from glob import glob
import json
import torch
import helpers
import numpy as np
import trimesh
import collections

def generate_seg_images(output_root_dir, rgb_folder, num_cam, img_name):
    """Generate black images PNG as everything is static currently. Images have the same size as the original images.
    TODO: change this to generate segmentation images when doing dynamic scenes.
    Parameters:
        rgb_folder (str): Path to the folder that contains rendering image.
        num_cam (str): Number of the camera.
        output_root_dir (str): Path to the output root folder.
    """
    original_image = PIL_Image.open(os.path.join(rgb_folder, img_name))
    width, height = original_image.size
    black_image = PIL_Image.new('L', (width, height), 0)
    seg_folder = os.path.join(output_root_dir, 'seg')
    target_seg_folder = os.path.join(seg_folder, num_cam)
    if not os.path.exists(target_seg_folder):
        os.makedirs(target_seg_folder)
    black_image.save(os.path.join(target_seg_folder,'render.png'))


def get_refnerf_blend_obj_paths():
    """Get all obj paths"""
    objs_dir = os.path.join(
        os.path.dirname(data.__file__),
        "refnerf-blend",
        'obj'
    )
    return glob(os.path.join(objs_dir, "*.obj"))


def filter_pointcloud(pts, w2c, ratio=0.001):
    """Filter pointcloud to reduce the number of points stored. Filtering is done by 
    randomly sampling 0.1% of the points, as well as by eliminating points that are
    too far from the camera.
    Parameters:
        pts (np.array): Pointcloud of the scene, shape [P, 3]
    Returns:
        pts (np.array): Filtered pointcloud of the scene, shape [Q, 3]"""
    # Remove points that are too far from the camera
    cam_centre = np.linalg.inv(w2c)[:3, 3]
    dist = np.linalg.norm(pts - cam_centre, axis=1)
    pts = pts[dist < 14.5]
    # Sample N points from the total pointcloud
    random_indices = np.random.choice(pts.shape[0], size=int(ratio * len(pts)), replace=False)
    pts = pts[random_indices]
    return pts


def load_meta_file(data_dir, flag):
    """Load the meta file from the specified directory.
    Parameters:
        data_dir (str): Absolute path to the object data directory, e.g. <path_to_data>/toaster
        flag (str): Flag to indicate the type of meta file to load: 'train', 'test'
    Return:
        meta (dict): Dictionary containing the meta data
    """
    if flag == 'train':
        file_name = 'train_meta.json'
    elif flag == 'test':
        file_name = 'test_meta.json'
    else:
        raise ValueError(f"Invalid flag: {flag}. Choose between 'train' and 'test'.")
    meta_path = os.path.join(data_dir, file_name)
    with open(meta_path, 'r') as file:
        meta = json.load(file)
    return meta


def load_rgb_gt(data_dir, partition):
    """Load the ground truth RGB images from a specified directory.
    Parameters:
        data_dir (str): Absolute path to the object data directory, e.g. <path_to_data>/toaster
        partition (str): Flag to indicate the type of data to load: 'train', 'test'
    Return:
        rgb_images (torch.Tensor): Images of shape [num_images, H, W, 3]
    """
    meta = load_meta_file(data_dir, partition)
    rgb_paths = [os.path.join(data_dir, 'ims', x) for x in meta['fn'][0]]

    # Convert image path to torch tensor
    rgb_images = torch.stack([helpers.load_rgb_image(x).permute(1, 2, 0) for x in rgb_paths],dim=0)[...,:3]
    return rgb_images


def load_depth_gt(data_dir, partition, is_disparity):
    """Load the disparity images from a selected directory and convert them to depth images.
    Parameters:
        data_dir (str): Absolute path to the object data directory, e.g. <path_to_data>/toaster
        partition (str): Flag to indicate the type of data to load: 'train', 'test'
    Return:
        depth_images (Tensor): Ground truth depth images, shape [num_images, H, W]
    """
    meta = load_meta_file(data_dir, partition)

    # Get the paths to the disparity images
    paths = [os.path.join(data_dir, 'depth', x.split(os.sep)[0], 'depth.tiff') for x in meta['fn'][0]]
        
    # self._debug_depth(depth_paths)
    if is_disparity:
        # Load disparity images. Check helpers.load_disparity_image() for more info about the conversion.
        disp_images = torch.stack([helpers.load_disparity_image(x) for x in paths], dim=0)

        # Convert disparity to depth
        depth_images = helpers.convert_disparity_to_depth(disp_images)
    else:
        # Load depth images
        depth_images = torch.stack([helpers.decode_image_to_tensor(x) for x in paths], dim=0)

    return depth_images


def load_pointcloud_gt(data_dir):
    """Load the ground truth pointcloud from the data directory.
    Parameters:
        data_dir (str): Absolute path to the object data directory, e.g. <path_to_data>/toaster
    Return:
        pointcloud (torch.Tensor): Ground truth pointcloud, shape [num_points, 3]
    """
    # Load the ground truth pointcloud
    pointcloud = np.load(os.path.join(data_dir, 'gt_pt_cld.npz'))['pts']
    pointcloud = torch.tensor(pointcloud).float().cuda()
    return pointcloud


def load_prediction(experiment_dir, pred_type=''):
    """Load the predicted data, which is stored as an .npz array. Convert them to torch tensor.
    Parameters:
        experiment_dir (str): Absolute path to the result directory, e.g. <path_to_output>/toaster_exp/toaster_15000
        pred_type (str): Type of data prediction to load: 'rgb', 'depth', 'pointcloud'
    Return:
        data (np.array): Predicted data
                             rgb: [num_images, H, W, 3],
                             depth: [num_images, H, W],
                             pointcloud: [num_points, 3]
    """
    if pred_type not in ['rgb', 'depth', 'pointcloud']:
        raise ValueError(f"Invalid pred_type: {pred_type}. Choose between 'rgb', 'depth', 'pointcloud'.")
    
    # Define a named tuple to store the file name and the key to access the data
    DataTuple = collections.namedtuple('DataTuple', ['file_name', 'key'])
    mapping = {
        'rgb': DataTuple('rgb_pred.npz', 'rgb'),
        'depth': DataTuple('depth_pred.npz', 'depth'),
        'pointcloud': DataTuple('pc_pred.npz', 'pts')
    }
    # Load data as numpy arrays
    data_pred_npy = np.load(os.path.join(experiment_dir, "eval", mapping[pred_type].file_name))[mapping[pred_type].key]
    
    return data_pred_npy


def get_points_in_bbox(
        points,
        min_coord=torch.tensor([-1.5, -1.5, -1]).cuda(),
        max_coord=torch.tensor([1.5, 1.5, 1]).cuda()
    ):
    """Get points that are inside a bounding box.
    Parameters:
        min_coord (torch.Tensor): Minimum coordinates of the bounding box, shape [3]
        max_coord (torch.Tensor)): Maximum coordinates of the bounding box, shape [3]
        points (torch.Tensor): Points to filter, shape [N, 3]
    Return:
        points_in_bbox (torch.Tensor): Points that are inside the bounding box, shape [M, 3]
    """
    mask = (points < max_coord).all(-1) & (points > min_coord).all(-1)
    points_in_bbox = points[mask]
    return points_in_bbox


def filter_dataset(N, ratio):
    """Filter the dataset by sampling a subset of the data at constant intervals.
    Parameters:
        N (int): Number of data points
        ratio (float): Ratio of the data to keep
    Return:
        final_indexes (list): Indexes of the data to keep
    """
    step = int(1 / ratio)
    final_indexes = np.arange(N * ratio) * step
    final_indexes = [int(i) for i in final_indexes]
    return final_indexes