"""
Script to convert the data from the RefNeRF dataset to the format used by this codebase.

Author: Mauro Comi, mauro.comi@bristol.ac.uk
Date: 28/07/2021
"""
import argparse
import os
import re
from glob import glob
import shutil
from PIL import Image
import json
import numpy as np
import data_making
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
from utils import utils_colmap, utils_data
import trimesh
import data
import helpers
import torch
import plotly.graph_objects as go

# Custom function to extract the integer following 'r_'
def _extract_number(file_path):
    match = re.search(r'r_(\d+)\.png', file_path)
    if match:
        return int(match.group(1))
    return 0  # Return 0 or some default value if pattern not found


# Custom function to extract the integer following 'r_'
def create_intrinsics_matrix(w, h, FOV):
    fx = w / (2 * np.tan(FOV / 2))
    k = np.array([[fx, 0, w/2], [0, fx, h/2], [0, 0, 1]])
    return k


def align_c2w(c2w):
    """Align the camera to world matrix to the convention used by this codebase.
    This consists in rotating the camera by 180 degrees around its X-axis."""
    # Extract rotation and translation components
    rotation_matrix = c2w[:3, :3]
    translation_vector = c2w[:3, 3]

    # Define the 3x3 rotation matrix for 180 degrees about the X-axis
    Rx_180 = np.array([[1, 0, 0],
                    [0, -1, 0],
                    [0, 0, -1]])

    # Apply the rotation
    rotated_matrix = rotation_matrix @ Rx_180

    # Recombine with the translation
    new_c2w = np.eye(4)  # Initialize a 4x4 identity matrix
    new_c2w[:3, :3] = rotated_matrix  # Set the top-left 3x3 part
    new_c2w[:3, 3] = translation_vector 

    return new_c2w


def process_metadata(metadata, sorted_ims, k):
    """Process the metadata to match the convention used by this codebase."""
    metadata = deepcopy(metadata) # Avoid modifying the original dictionary for safety
    metadata['k'] = np.tile(k, (1, len(sorted_ims), 1, 1)).tolist()
    metadata['w2c'] = np.array(metadata['w2c'])[None, :].tolist() # Add a dimension
    metadata['fn'] = np.array(metadata['fn'])[None, :].tolist()  # Add a dimension
    metadata['cam_id'] = np.array(metadata['cam_id'])[None, :].tolist()  # Add a dimension
    return metadata

def get_sorted_ims_path(input_dir, regex_pattern):
    # Get images paths
    ims_path = glob(os.path.join(input_dir, "r_?*.png"))
    # Filter the files using the regex pattern
    ims_path = [f for f in ims_path if re.search(regex_pattern, os.path.basename(f))]
    return sorted(ims_path, key=_extract_number)


def remove_temporary_files(sparse_init_dir, sparse_bin_dir):
    """Remove temporary and unused files and directories"""
    shutil.rmtree(sparse_init_dir)
    shutil.rmtree(sparse_bin_dir)


def extract_colmap_feature(output_obj_dir):
    convert_py_path = os.path.join(os.path.dirname(data_making.__file__), "convert.py")
    convert_cmd = f"python {convert_py_path} "\
        f"-s {output_obj_dir} "\
        f"--camera SIMPLE_PINHOLE"
    exit_code = os.system(convert_cmd)
    if exit_code != 0:
        print(f"Conversion failed with code {exit_code}. Exiting.")
        exit(exit_code)


def convert_bin_to_txt(output_obj_dir):
    # Set directories
    sparse_dir = os.path.join(output_obj_dir, "sparse_bin")
    sparse_txt_dir = os.path.join(output_obj_dir, "sparse")
    helpers.create_dirs(sparse_txt_dir)
    # Convert points3D.bin to .txt
    bin_to_txt_cmd = "colmap model_converter "\
        f"--input_path  {sparse_dir} "\
        f"--output_path {sparse_txt_dir} "\
        "--output_type TXT"
    exit_code = os.system(bin_to_txt_cmd)
    if exit_code != 0:
        print(f"Conversion failed with code {exit_code}. Exiting.")
        exit(exit_code)


def extract_xyzrgb_from_txt(file_path):
    xyzrgb = []
    with open(file_path, 'r') as file:
        for line in file:
            # Skip lines that don't start with a point ID
            if not line.strip() or line.startswith('#'):
                continue

            # Split the line and extract X, Y, Z coordinates
            parts = line.split()
            x, y, z, r, g, b = map(float, parts[1:7])
            r, g, b = r/255., g/255., b/255.
            xyzrgb.append([x, y, z, r, g, b, 1])  # Add a 1 for segmentation
    return np.array(xyzrgb)


def w2c_to_pose(w2c):
    """Convert the w2c to a pose (quaternion, translation) tuple.
    Parameters:
        w2c: list of 4x4 w2c matrix
    Returns:
        quaternion: list of quaternions, IMPORTANT: convention is xyzw
        translation: list of translations"""
    w2c = np.array(w2c)
    # Extract rotation and translation
    rotation_matrix = w2c[:3, :3]
    translation_vector = w2c[:3, 3]
    # Convert rotation matrix to quaternion (xyzw)
    quaternion = R.from_matrix(rotation_matrix).as_quat()
    return quaternion, translation_vector


def create_images_txt(meta, j):
    """Create the images.txt file required by COLMAP: https://colmap.github.io/format.html#images-txt
    The required format is ID, QW, QX, QY, QZ, TX, TY, TZ, camera ID, image_name"""
    images_content = ""
    for i, w2c in enumerate(meta['w2c'][0]):
        quaternion, translation = w2c_to_pose(w2c)
        # Format for COLMAP (ID, quaternion, translation, camera ID, image name)
        images_content += f"{i+1+j} {quaternion[3]} {quaternion[0]} {quaternion[1]} {quaternion[2]} {translation[0]} {translation[1]} {translation[2]} 1 {i+j}/render.png\n\n"
        # images_content += f"{i+1+j} {quaternion[3]} {quaternion[0]} {quaternion[1]} {quaternion[2]} {translation[0]} {translation[1]} {translation[2]} 1 image_{i+j}.png\n\n"
    
    return images_content


def create_database(output_obj_dir):
    """Create the database required by COLMAP."""
    database_path = os.path.join(output_obj_dir, "database.db")
    if os.path.exists(database_path):
        os.remove(database_path)
    database_cmd = f"colmap database_creator --database_path {database_path}"
    exit_code = os.system(database_cmd)
    if exit_code != 0:
        print(f"Error in database creation: {exit_code}. Exiting.")
        exit(exit_code)
    return database_path


def populate_database(database_path, meta_train, meta_test, k):
    """Populate the database with the images and corresponding metadata."""
    db = utils_colmap.COLMAPDatabase.connect(database_path)

    db.create_tables()

    fns = meta_train['fn'][0] + meta_test['fn'][0]
    w2cs = meta_train['w2c'][0] + meta_test['w2c'][0]
    model, width, height, params = (
        0,
        meta_train['w'],
        meta_train['h'],
        np.array((k[0][0], k[0][2], k[1][2]))
    )
    for i in range(len(meta_train['fn'][0] + meta_test['fn'][0])):
        quaternion, translation = w2c_to_pose(w2cs[i])
        camera_id = db.add_camera(model, width, height, params)
        image_id = db.add_image(
            fns[i], camera_id,
            np.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]]),
            np.array([translation[0], translation[1], translation[2]])
        )
    db.commit()
    db.close()


def create_temporary_dirs_files(output_obj_dir, meta_train, meta_test, sorted_ims_train, k):
    """Create the temporary directories and files required by COLMAP: images.txt, cameras.txt, and empty points3D.txt"""
    sparse_init_dir = os.path.join(output_obj_dir, "sparse_init")
    sparse_bin_dir = os.path.join(output_obj_dir, "sparse_bin")
    helpers.create_dirs([sparse_init_dir, sparse_bin_dir])

    # Create images.txt content
    images_content = ""
    images_content += create_images_txt(meta_train, j=0)
    images_content += create_images_txt(meta_test, j=len(sorted_ims_train))
    # Save images.txt
    images_path = os.path.join(sparse_init_dir, "images.txt")
    with open(images_path, "w") as file:
        file.write(images_content)

    # Write cameras.txt (example with dummy intrinsics)
    cameras_path = os.path.join(sparse_init_dir, "cameras.txt")
    with open(cameras_path, "w") as file:
        # cameras.txt: CAMERA_ID MODEL W H Fx Cx Cy
        file.write(f"1 SIMPLE_PINHOLE {meta_train['w']} {meta_train['h']} {k[0][0]} {k[0][2]} {k[1][2]}\n")  

    # Write empty points3D.txt
    points3D_path = os.path.join(sparse_init_dir, "points3D.txt")
    with open(points3D_path, "w") as file:
        file.write("")

    return sparse_init_dir, sparse_bin_dir


def feature_extractor(database_path, output_obj_dir):
    """Run COLMAP's feature extractor."""
    feature_extractor_cmd = f"colmap feature_extractor " \
    f"--database_path {database_path} " \
    f"--image_path {output_obj_dir}/ims"
    exit_code = os.system(feature_extractor_cmd)
    if exit_code != 0:
        print(f"Feature extractor failed with code {exit_code}. Exiting.")
        exit(exit_code)
    else:
        print("Feature extractor finished.")


def exhaustive_matcher(database_path):
    """Run COLMAP's exhaustive matcher."""
    exhaustive_matcher_cmd = "colmap exhaustive_matcher "\
        f"--database_path {database_path}"
    exit_code = os.system(exhaustive_matcher_cmd)
    if exit_code != 0:
        print(f"Exhaustive matcher failed with code {exit_code}. Exiting.")
        exit(exit_code)
    else:  
        print("Exhaustive matcher finished.")


def point_triangulator(output_obj_dir, database_path, sparse_init_dir, sparse_bin_dir):
    # Run COLMAP's point triangulator using the known poses
    point_triangulator_cmd = "colmap point_triangulator "\
        f"--database_path {database_path} "\
        f"--image_path {output_obj_dir}/ims "\
        f"--input_path {sparse_init_dir} "\
        f"--output_path {sparse_bin_dir}"
    exit_code = os.system(point_triangulator_cmd)
    if exit_code != 0:
        print(f"Point triangulator failed with code {exit_code}. Exiting.")
        exit(exit_code)
    else:
        print("Point triangulator finished.")


def extract_pointcloud_gt(output_obj_dir, dataset):
    """Extract the ground truth point cloud from the disparity images in the training set.
    Then, save the point cloud in a .npz file.
    The reason why the point cloud is not extracted directly from the object mesh is that internal
    structures are not visible from the outside. Extracting ground truth point cloud from disparity images
    reflects the same visible surface information that would be captured by our reconstruction
    Parameters:

    Returns:

    """
    if dataset == 'ShinyBlender':
        is_disparity = True
    elif dataset == 'GlossySynthetic':
        is_disparity = False
    else:
        raise ValueError('Invalid dataset name')
    # Set paths
    pc_cld_path = os.path.join(output_obj_dir, 'gt_pt_cld.npz')

    # Load files
    meta_train = utils_data.load_meta_file(output_obj_dir, 'train')
    depth_images = utils_data.load_depth_gt(output_obj_dir, 'train', is_disparity)

    # Defines variables
    w, h = meta_train['w'], meta_train['h']
    w2c_all = torch.tensor(meta_train['w2c'][0]).cuda().float()
    k_all = torch.tensor(meta_train['k'][0]).cuda().float()
    inv_w2c_all = torch.linalg.inv(w2c_all)     # [I, 4, 4]
    inv_k_all = torch.linalg.inv(k_all)         # [I, 3, 3]
    def_pix = torch.tensor(
        np.stack(np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5, 1), -1).reshape(-1, 3)
    ).cuda().float()  # [N, 3], where N=num_pixel
    pointcloud_all = [] 

    # Loop over the images
    for idx in range(w2c_all.shape[0]):
        # Calculate pixel coordinates in camera space
        inv_k_cam = inv_k_all[idx]         # [3, 3]
        inv_w2c_cam = inv_w2c_all[idx]     # [4, 4]
        p_cam = (inv_k_cam @ def_pix.permute(1,0)).permute(1,0)    # [N, 3]
        
        # Filter out pixels with depth=0
        depth_1d = torch.tensor(depth_images[idx]).cuda().view(-1) # all depth values in a single tensor, shape [M]
        valid_depth_mask = depth_1d != 0
        p_cam = p_cam[valid_depth_mask] # remove pixels with depth=0
        depth_1d = depth_1d[valid_depth_mask]

        # Randomly sample 5000 points to reduce number of points stored
        indexes = torch.randperm(p_cam.shape[0])
        p_cam = p_cam[indexes][:5000]
        depth_1d = depth_1d[indexes][:5000]

        # Calculate 3D points in world frame
        depth_cam = p_cam * depth_1d.unsqueeze(1)
        depth_cam = torch.cat([depth_cam, torch.ones(5000,1).cuda()], dim=1) 
        p_w = inv_w2c_cam @ depth_cam.permute(1,0)
        p_w = p_w.permute(1,0)[:, :3]
        p_w_cpu = utils_data.filter_pointcloud(
            p_w.cpu().numpy(),
            w2c_all[idx].cpu().numpy(),
            ratio_pointcloud=1
        )
        
        pointcloud_all.extend(p_w_cpu)

    pointcloud_all = np.array(pointcloud_all, dtype=np.float32)
    np.savez_compressed(pc_cld_path, pts=pointcloud_all)
        

def main(args):

    if args.dataset == 'ShinyBlender':
        input_dir = 'data/refnerf'
        dataset_dir = 'shiny-blender-3DGS'

    elif args.dataset == 'GlossySynthetic':
        input_dir = 'data/glossy-synthetic-Blender'
        dataset_dir = 'glossy-synthetic-3DGS'

    else:
        raise ValueError(f"Invalid dataset: {args.dataset}.")
  
    objects = os.listdir(input_dir)

    for obj in objects:   # e.g. obj = "toaster"
    
        input_obj_dir = os.path.join(input_dir, obj)

        output_obj_dir = os.path.join(args.output_dir, dataset_dir, obj)    # e.g. output_obj_dir = "data/shiny-blender-3DGS/toaster"

        # Read RefNeRF json files
        json_path_train = os.path.join(input_obj_dir, "transforms_train.json")
        json_path_test = os.path.join(input_obj_dir, "transforms_test.json")
        with open(json_path_train, "r") as f:
            json_train = json.load(f)
        with open(json_path_test, "r") as f:
            json_test = json.load(f)

        regex_pattern = r'^r_\d+\.png$'   # e.g. r_*.png where * is a number
        # Sort the list using the custom key function
        sorted_ims_train = get_sorted_ims_path(
            os.path.join(input_obj_dir, "train"), regex_pattern)
        sorted_ims_test = get_sorted_ims_path(
            os.path.join(input_obj_dir, "test"), regex_pattern)
        sorted_ims = sorted_ims_train + sorted_ims_test

        # Initialise meta data ['w', 'h', 'k', 'w2c', 'fn', 'cam_id']
        # Read H and W from first image in sorted_ims_train
        image0 = Image.open(sorted_ims_train[0])
        h_for_image, w_from_image = np.asarray(image0).shape[0], np.asarray(image0).shape[1] 
        meta_train = {'w': w_from_image, 'h': h_for_image, 'k': None, 'w2c': [], 'fn': [], 'cam_id': []}
        meta_test = {'w': w_from_image, 'h': h_for_image, 'k': None, 'w2c': [], 'fn': [], 'cam_id': []}

        k = create_intrinsics_matrix(
            meta_train['w'],
            meta_train['h'],
            json_train['camera_angle_x']
        )
    
        for i in range(0, len(sorted_ims)):
            # Create image directory 
            cam_image_dir = os.path.join(output_obj_dir, "ims", str(i))
            depth_image_dir = os.path.join(output_obj_dir, "depth", str(i))
            seg_image_dir = os.path.join(output_obj_dir, "seg", str(i))
            helpers.create_dirs([cam_image_dir, depth_image_dir, seg_image_dir])

            # Convert image PNG to JPEG and save it
            shutil.copy2(sorted_ims[i], os.path.join(cam_image_dir, "render.png"))

            # Copy depth images
            if args.dataset == 'ShinyBlender':
                depth_path = sorted_ims[i].replace(".png", "_disp.tiff")
            elif args.dataset == 'GlossySynthetic':
                depth_path = sorted_ims[i].replace(".png", "_depth.tiff")
            else:
                raise ValueError(f"Invalid dataset: {args.dataset}.")
            if os.path.exists(depth_path):
                shutil.copy(depth_path, os.path.join(depth_image_dir, "depth.tiff"))

            # Generate black images as everything is static currently. Images have the same size as the original images
            # Segmentation imaged need to be .PNG
            utils_data.generate_seg_images(
                output_obj_dir,
                cam_image_dir,
                str(i),
                img_name='render.png'
            )

            # Create metadata file
            meta_results = meta_train if i < len(sorted_ims_train) else meta_test
            json_file = json_train if i < len(sorted_ims_train) else json_test
            j = i if i < len(sorted_ims_train) else i - len(sorted_ims_train)

            c2w = np.array(json_file['frames'][j]['transform_matrix'])
            c2w = align_c2w(c2w)  # Align c2w to the convention used by this codebase
            w2c = np.linalg.inv(c2w)
            meta_results['w2c'].append(w2c)
            meta_results['fn'].append(os.path.join(os.path.basename(cam_image_dir), "render.png"))
            meta_results['cam_id'].append(os.path.basename(cam_image_dir))

        # Postprocess metadata
        meta_train = process_metadata(meta_train, sorted_ims_train, k)
        meta_test = process_metadata(meta_test, sorted_ims_test, k)

        # Save metadata
        meta_path_train = os.path.join(output_obj_dir, "train_meta.json")
        meta_path_test = os.path.join(output_obj_dir, "test_meta.json")
        with open(meta_path_train, 'w') as f:
            json.dump(meta_train, f)
        with open(meta_path_test, 'w') as f:
            json.dump(meta_test, f)

        # Create temporary colmap directories
        sparse_init_dir, sparse_bin_dir = create_temporary_dirs_files(output_obj_dir, meta_train, meta_test, sorted_ims_train, k)
        
        # Create database
        database_path = create_database(output_obj_dir)
        
        # Populate database with known camera poses
        populate_database(
            f'{database_path}',
            meta_train,
            meta_test,
            k
        )

        # Run COLMAP's feature extractor
        feature_extractor(database_path, output_obj_dir)

        # Run COLMAP's exhaustive matcher
        exhaustive_matcher(database_path)

        # Run COLMAP's point triangulator using the known poses
        point_triangulator(output_obj_dir, database_path, sparse_init_dir, sparse_bin_dir)
        
        # Convert points3D.bin to .txt
        convert_bin_to_txt(output_obj_dir)

        # Convert points3D.txt to .npz
        init_pt_cld = extract_xyzrgb_from_txt(os.path.join(output_obj_dir, "sparse", "points3D.txt"))
        np.savez(os.path.join(output_obj_dir, "init_pt_cld.npz"), data=init_pt_cld)

        # Extract ground truth point cloud
        if args.dataset == 'ShinyBlender':
            output_seq_dir = os.path.join(os.path.dirname(data.__file__), obj)
            extract_pointcloud_gt(output_obj_dir, dataset=args.dataset)
        elif args.dataset == 'GlossySynthetic':
            shutil.copy(os.path.join(input_obj_dir, 'gt_pt_cld.npz'), os.path.join(output_obj_dir, "gt_pt_cld.npz"))

        # Remove unnecessary files and directories
        remove_temporary_files(sparse_init_dir, sparse_bin_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='', help="Choose between 'ShinyBlender' or 'GlossySynthetic'")
    parser.add_argument("--output_dir", type=str, default="data/", help="Path to the output directory.")
    args = parser.parse_args()    

    main(args) 