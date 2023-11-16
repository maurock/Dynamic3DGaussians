"""This script converts data extracted from COLMAP to the required data for Dynamic 3D Gaussians."""
import argparse
import os
from glob import glob
import shutil
import numpy as np
from utils import utils_colmap
import json
from PIL import Image as PIL_Image
from utils import utils_data_making

def get_intrinsics_from_txt(path):
    """Convert the file `cameras.txt` extracted from colmap (SIMPLE_PINHOLE) to camera intrinsics."""
    ks = []
    with open(path, 'r') as f:
        for idx, line in enumerate(f.readlines()):
            # Skip first lines
            if idx < 3:
                continue
            line = line.strip().split(' ')

            # Convert x,y,z,r,g,b values from str to float
            for i in range(2, 7):
                line[i] = float(line[i])

            w = line[2]
            h = line[3]
            fx = line[4]
            fy = line[4]
            cx = line[5]
            cy = line[6]

            k = [ [fx, 0, cx], [0, fy, cy], [0, 0, 1] ]
            ks.append(k)
    
    return ks


def get_pt_cld_from_text(path):
    """Generate init_pt_cld.npz: shape (N, 7) where N is the number of points"""

    pt_cld = []

    with open(path, 'r') as f:

        for idx, line in enumerate(f.readlines()):
            # Skip first lines
            if idx < 3:
                continue
            line = line.strip().split(' ')
            
            # Convert x,y,z,r,g,b values from str to float
            for i in range(1, 7):
                line[i] = float(line[i])
            
            
            xyz = [line[1], line[2], line[3]]
            rgb = [line[4]/255.0, line[5]/255.0, line[6]/255.0]
            seg = [0.0]   # Always static for now 
            
            pt_cld.extend([xyz + rgb + seg])
            
    pt_cld = np.array(pt_cld)

    return pt_cld


def main(args):
    print('Converting COLMAP data to Dynamic 3D Gaussians data...')

    # Copy images from COLMAP folder to data. Images need to be .JPEG.
    img_source_folder = os.path.join(args.colmap_path, 'images')
    ims_folder = os.path.join(args.output_path, args.dataset_name, 'ims')

    # If images are not in an individual folder per camera, create that. This is usually ther case
    # for images taken with a phone/camera, not with a 3D renderer (which has store data in a folder per camera)
    if len(glob(os.path.join(img_source_folder, '*', '*.jpg'))) == 0:
        num_cam = '0'
        for img_source_file in glob(os.path.join(img_source_folder, '*.jpg')):
            target_folder = os.path.join(ims_folder, num_cam)
            img_name = img_source_file.split('/')[-1]
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)
            shutil.copy(img_source_file, os.path.join(target_folder, img_name))

            # Generate black images as everything is static currently. Images have the same size as the original images
            # Segmentation imaged need to be .PNG
            # TODO: change this to generate the segmentation images
            utils_data_making.generate_seg_images(args, target_folder, num_cam, img_name)
            num_cam = str(eval(num_cam) + 1)
    else:
        # This is the case for images taken with a 3D renderer. Images are already in a folder per camera
        img_source_files = glob(os.path.join(img_source_folder, '*', '*.jpg'))
        for img_source_file in img_source_files:
            num_cam = img_source_file.split('/')[-2]
            target_folder = os.path.join(ims_folder, num_cam)
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)
            shutil.copy(img_source_file, os.path.join(target_folder, 'render.jpg'))

            # Generate black images as everything is static currently. Images have the same size as the original images
            # Segmentation imaged need to be .PNG
            # TODO: change this to generate the segmentation images
            utils_data_making.generate_seg_images(args, target_folder, num_cam, img_name='render.jpg')
    
    # Generate init_pt_cld.npz: shape (N, 7) where N is the number of points
    pt_cld_path = os.path.join(args.colmap_path, 'sparse', '0', 'points3D.bin')
    xyzs, rgbs, _ = utils_colmap.read_points3D_binary(pt_cld_path)
    seg = np.ones_like(xyzs[:, 0])[:, None]   # Always static for now, segmentation always 1
    rgbs = rgbs / 255.0
    pt_cld = dict()
    pt_cld = np.concatenate((xyzs, rgbs, seg), axis=1).tolist()
    if not os.path.exists(os.path.join(args.output_path, args.dataset_name)):
        os.makedirs(os.path.join(args.output_path, args.dataset_name))
    np.savez(os.path.join(args.output_path, args.dataset_name, 'init_pt_cld.npz'), data=pt_cld)
    print('Point cloud saved')

    # Get intrinsics and extrinsics values from COLMAP
    data = dict()
    extrinsics_path = os.path.join(args.colmap_path, 'sparse', '0', 'images.bin')
    intrinsics_path = os.path.join(args.colmap_path, 'sparse', '0', 'cameras.bin')
    extr = utils_colmap.read_extrinsics_binary(extrinsics_path)  # w2c
    intr = utils_colmap.read_intrinsics_binary(intrinsics_path)

    data['w'] = intr[1].width
    data['h'] = intr[1].height
    
    # Generate intrinsics (N, 3, 3) where N is the number of unique cameras
    k = utils_colmap.get_intrinsics_matrix(extr, intr) 
    data['k'] = [k] # Add dimension as I only have 1 timestamp for now 
    print('Intrinsics matrix calculated')

    # Generate extrinsics (N, 4, 4) where N is the number of unique cameras
    w2c = utils_colmap.get_extrinsics_matrix(ims_folder, extr, intr) 
    data['w2c'] = [w2c] # Add dimension as I only have 1 timestamp for now   
    print('Extrinsics matrix calculated')

    # Get images
    fn, cam_id = utils_colmap.get_cam_images(extr)
    data['fn'] = [fn] # Add dimension as I only have 1 timestamp for now 
    data['cam_id'] = [cam_id]

    # Save data as a json file
    with open(os.path.join(args.output_path, args.dataset_name, 'train_meta.json'), 'w') as f:
        json.dump(data, f)

if __name__=='__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--colmap_path', type=str, default='', help='Path to the COLMAP data.')
    args.add_argument('--output_path', type=str, default='data/', help='Path to the output data.')
    args.add_argument('--dataset_name', type=str, default='', help='Dataset name.')

    args = args.parse_args()

    main(args)