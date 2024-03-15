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
import data_making.refnerf_to_data as refnerf_to_data


def main():
    output_obj_dir = os.path.join(
        os.path.dirname(data.__file__), 'real', 'toaster_colmap_input'
    )

    meta_train = utils_data.load_meta_file(output_obj_dir, 'train')
    meta_test = utils_data.load_meta_file(output_obj_dir, 'test')

    k = np.array(meta_train['k'][0][0])

    # Function to extract the numerical part of the filename for sorting
    def sort_key(filename):
        # This regex pattern finds the digits in the filename
        match = re.search(r'(\d+)', filename)
        if match:
            return int(match.group(1))
        return 0

    # Retrieve image file paths and sort them
    full_paths = glob(os.path.join(output_obj_dir, 'ims', 'image_*.png'))
    image_names = [_.split('/')[-1] for _ in full_paths]
    sorted_ims_train = sorted(image_names, key=sort_key)

    # Create temporary colmap directories
    sparse_init_dir, sparse_bin_dir = refnerf_to_data.create_temporary_dirs_files(output_obj_dir, meta_train, meta_test, sorted_ims_train, k)

    # Create database
    database_path = refnerf_to_data.create_database(output_obj_dir)

    # Populate database with known camera poses
    refnerf_to_data.populate_database(
        f'{database_path}',
        meta_train,
        meta_test,
        k
    )

    # Run COLMAP's feature extractor
    refnerf_to_data.feature_extractor(database_path, output_obj_dir)

    # Run COLMAP's exhaustive matcher
    refnerf_to_data.exhaustive_matcher(database_path)

    # Run COLMAP's point triangulator using the known poses
    refnerf_to_data.point_triangulator(output_obj_dir, database_path, sparse_init_dir, sparse_bin_dir)

    # Convert points3D.bin to .txt
    refnerf_to_data.convert_bin_to_txt(output_obj_dir)

    # Convert points3D.txt to .npz
    init_pt_cld = refnerf_to_data.extract_xyzrgb_from_txt(os.path.join(output_obj_dir, "sparse", "points3D.txt"))
    np.savez(os.path.join(output_obj_dir, "init_pt_cld.npz"), data=init_pt_cld)


if __name__ == '__main__':
    main()