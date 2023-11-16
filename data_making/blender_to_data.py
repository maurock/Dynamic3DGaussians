import argparse
import os
from glob import glob
import shutil
from utils import utils_data_making, utils_colmap
import numpy as np
import json

def get_intrinsics(camera_info):
    """Get intrisics matrix from camera info"""
    fx = camera_info['fx']
    fy = camera_info['fy']
    cx = camera_info['width'] / 2
    cy = camera_info['height'] / 2
    position = camera_info['position']
    k = [[fx, 0, cx],
         [0, fy, cy],
         [0, 0, 1]]
    
    return k      

def main(args):
    print('Converting Blender data to Dynamic 3D Gaussians data...')

    # Copy images from bleder folder to data. Images need to be .JPEG.
    img_source_folder = os.path.join(args.data_path, 'ims')
    ims_folder = os.path.join(args.output_path, 'ims')
    img_source_files = glob(os.path.join(img_source_folder, '*', '*.jpg'))
    for img_source_file in img_source_files:
        num_cam = img_source_file.split('/')[-2]
        target_folder = os.path.join(ims_folder, num_cam)
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        shutil.copy(img_source_file, os.path.join(target_folder, 'render.jpg'))

        # Generate black images as everything is static currently. Images have the same size as the original images
        # Segmentation imaged need to be .PNG
        # TODO: change this to generate real segmentation images
        utils_data_making.generate_seg_images(args, target_folder, num_cam, img_name='render.jpg')

    # Copy point cloud from Blender folder to repository
    pc_source_folder = os.path.join(args.data_path, 'init_pt_cld.npz')
    shutil.copy(pc_source_folder, os.path.join(args.output_path, 'init_pt_cld.npz'))

    # Get intrinsics and extrinsics values from Blender data
    data = dict()
    cameras_info_path = os.path.join(args.data_path, 'cameras_gt.json')

    # read cameras info
    with open(cameras_info_path, 'r') as f:
        cameras_info = json.load(f)  # list of dictionaries, each dict is a camera

    data['w'] = cameras_info[0]['width']
    data['h'] = cameras_info[0]['height']
    
    w2c, k, cam_id, fn = [], [], [], []
    for i in range(len(cameras_info)):
        w2c.append(cameras_info[i]['w2c'])
        k.append(get_intrinsics(cameras_info[i]))
        cam_id.append(str(cameras_info[i]['id']))
        fn.append(f"{cameras_info[i]['id']}/render.jpg")

    # IMPOTANT! Change this when moving from static to dynamic
    data['w2c'] = [w2c]
    data['k'] = [k]
    data['cam_id'] = [cam_id]
    data['fn'] = [fn]
        
    # Save data as a json file
    with open(os.path.join(args.output_path, 'train_meta.json'), 'w') as f:
        json.dump(data, f)


if __name__=='__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--data_path', type=str, default='', help='Path to the Blender data.')
    args.add_argument('--output_path', type=str, default='data/YOUR_DATASET', help='Path to the output data.')

    args = args.parse_args()

    main(args)