"""This script converts data extracted from COLMAP to the required data for Dynamic 3D Gaussians."""
import argparse
import os
from glob import glob
import shutil
import numpy as np

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

    # Copy images
    img_source_folder = os.path.join(args.colmap_path, 'input')
    img_target_folder = os.path.join(args.output_path, args.dataset_name, 'ims')

    if not os.path.exists(img_target_folder):
        os.makedirs(img_target_folder)
    
    img_files = glob(os.path.join(img_source_folder, '*.png'))
    for img_file in img_files:
        shutil.copy(img_file, img_target_folder)
    
    # Generate init_pt_cld.npz: shape (N, 7) where N is the number of points
    # Read point cloud (txt) from COLMAP
    pt_cld_path = os.path.join(args.colmap_path, 'sparse', '0_text', 'points3D.txt')
    pt_cld = get_pt_cld_from_text(pt_cld_path)    
    np.savez(os.path.join(args.output_path, args.dataset_name, 'init_pt_cld.npz'), pt_cld)

    # Generate intrinsics (N, 3, 3) where N is the number of unique cameras
    intrinsics_path = os.path.join(args.colmap_path, 'sparse', '0_text', 'cameras.txt')
    ks = [get_intrinsics_from_txt(intrinsics_path)] # Add dimension as I only have 1 timestamp for now    


if __name__=='__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--colmap_path', type=str, default='', help='Path to the COLMAP data.')
    args.add_argument('--output_path', type=str, default='data/', help='Path to the output data.')
    args.add_argument('--dataset_name', type=str, default='', help='Dataset name.')

    args = args.parse_args()
    main(args)