"""When recording images, the results in folder structure is not the one expected by this repo.
In this script, I simply add the camera index per image, storing each index inside the corresponding folder."""

import os
import shutil
from glob import glob
import argparse

def main(args):
    img_source_folder = os.path.join(args.data_path, 'input')
    
    img_source_names = os.listdir(img_source_folder)
    
    num_cam = '0'

    for img_source_name in img_source_names:
        img_source_file = os.path.join(img_source_folder, img_source_name)

        target_folder = os.path.join(img_source_folder, num_cam)

        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        shutil.copy(img_source_file, os.path.join(target_folder, 'render.jpg'))

        num_cam = str(eval(num_cam) + 1)

        # remove the original image
        os.remove(img_source_file)

if __name__=='__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--data_path', type=str, default='', help='Path to the folder containing data.')
    args = args.parse_args()

    args.data_path = '/home/mauro/Downloads/gise_resized/'
    main(args)