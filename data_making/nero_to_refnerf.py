import os
import numpy as np
import math
import json
import glob
import argparse
import pickle
import shutil
from skimage.io import imread, imsave
import data
import helpers
from PIL import Image
from plyfile import PlyData

def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    # Set folders
    output_dir = os.path.join(os.path.dirname(data.__file__), 'glossy-synthetic-Blender')
    train_dir = os.path.join(os.path.dirname(data.__file__), 'GlossySynthetic')
    test_dir = os.path.join(os.path.dirname(data.__file__), 'glossy-synthetic-nvs')

    train_scenes = os.listdir(train_dir)
    test_scenes = [name + '_nvs' for name in os.listdir(train_dir)]

    # TRAIN
    splits = ['train', 'test']
    for split in splits:

        scenes = train_scenes if split == 'train' else test_scenes

        for idx, scene in enumerate(scenes):

            print(f'[INFO] Process scene {scene}') 

            # Set split dir
            split_dir = train_dir if split == 'train' else test_dir

            scene_output = scene if split == 'train' else scene.split('_')[0]

            # Set input and output dirs
            input_scene_dir = os.path.join(split_dir, scene)           # .../GlossySynthetic/cat, .../glossy-synthetic-nvs/cat_nvs
            output_scene_dir = os.path.join(output_dir, scene_output)         # .../glossy-synthetic-3DGS/cat/
            output_split_dir = os.path.join(output_dir, scene_output, split)       # .../glossy-synthetic-3DGS/cat/train
            helpers.create_dirs(output_split_dir)

            # Set img ids
            img_num = len(glob.glob(f'{input_scene_dir}/*.pkl'))
            img_ids = [k for k in range(img_num)] 

            # Read data
            cams = [read_pickle(f'{input_scene_dir}/{k}-camera.pkl') for k in range(img_num)]  # pose(3,4)  K(3,3)

            # Set output names for images and depth images
            output_img_files = [f'{output_split_dir}/r_{k}.png' for k in img_ids]     # output
            # if split == 'train':
            #     depth_files = [f'{input_scene_dir}/{k + buffer}-depth.png' for k in range(img_num)]
            # else:
            #     depth_files = [f'{input_scene_dir}/{k + buffer}-depth0001.exr' for k in range(img_num)]

            #test_ids, train_ids = read_pickle(os.path.join(opt.path, 'synthetic_split_128.pkl'))

            frames = []
            for image, cam in zip(output_img_files, cams):
                w2c = np.array(cam[0].tolist()+[[0,0,0,1]])
                c2w = np.linalg.inv(w2c)
                c2w[:3, 1:3] *= -1  # opencv -> blender/opengl
                frames.append({
                    'file_path': os.path.join(f'./{split}/{os.path.basename(image)}'),
                    'transform_matrix': c2w.tolist(),
                })

            fl_x = float(cams[0][1][0,0])
            fl_y = float(cams[0][1][1,1])

            camera_angle_x = 2 * math.atan(800 / (2 * fl_x))

            transforms = {
                'camera_angle_x': camera_angle_x,
                'frames': frames
            }          

            # write json
            json_out_path = os.path.join(output_scene_dir, f'transforms_{split}.json')
            print(f'[INFO] write to {json_out_path}')
            with open(json_out_path, 'w') as f:
                json.dump(transforms, f, indent=2)
            
            # write imgs
            print(f'[INFO] Process rgbs')
            print(f'[INFO] write to {output_split_dir}')
            for img_id in img_ids:     # index
                if split == 'train':
                    depth_img = imread(f'{input_scene_dir}/{img_id}-depth.png')
                    depth = depth_img.astype(np.float32) / 65535    # [H, W, 3]
                else:
                    depth_img = helpers.decode_image_to_tensor(f'{input_scene_dir}/{img_id}-depth0001.exr').cpu().numpy()  # (H, W, 3)
                    depth = depth_img[...,0]

                mask = depth < 1         # [H, W] between 0 and 1
                mask = mask[...,None]    # [H, W, 1]
                mask_255 = (mask * 255).astype(np.uint8)  # [H, W, 1] between 0 and 255
 
                image = imread(f'{input_scene_dir}/{img_id}.png')[...,:3]
                # Process image: remove background based on depth (mask)
                image = image * mask
                image = np.concatenate([image, mask_255], axis=-1)

                imsave(f'{output_split_dir}/r_{img_id}.png', image)
                imsave(f'{output_split_dir}/r_{img_id}_depth.tiff', depth)

            # Save ply as .npz - only for training, as the path doesnt exist in the test dataset
            points_file = os.path.join(input_scene_dir, "eval_pts.ply")
            if os.path.exists(points_file):
                pc_cld_path = os.path.join(output_scene_dir, 'gt_pt_cld.npz')

                with open(points_file, 'rb') as f:
                    plydata = PlyData.read(f)

                x, y, z = plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']
                pointcloud_all = np.array(list(zip(x, y, z)))

                np.savez_compressed(pc_cld_path, pts=pointcloud_all)

            print(f"[INFO] Scene {scene} processed.")