# Dynamic 3D Gaussians
This repository is based on the brilliant [Dynamic 3D Gaussians:
Tracking by Persistent Dynamic View Synthesis](https://dynamic3dgaussians.github.io/), which extends Gaussian Splatting to dynamic scenes, with accurate novel-view synthesis and dense 3D 6-DOF tracking.<br><br>

Differently form the original code, this repository provides:
- Code and intructions for data extraction using Blender. [Go to the instructions](#)
- A better Visualiser tool, with interactive commands to iterate through known camera poses and change mode ('colors', 'depth').
- Minor changes and addition of comments.

Please consider citing the official implementation:

```
@inproceedings{luiten2023dynamic,
  title={Dynamic 3D Gaussians: Tracking by Persistent Dynamic View Synthesis},
  author={Luiten, Jonathon and Kopanas, Georgios and Leibe, Bastian and Ramanan, Deva},
  booktitle={3DV},
  year={2024}
}
```

## Installation (original repo)
```bash
# Install this repo (pytorch)
git clone git@github.com:JonathonLuiten/Dynamic3DGaussians.git
conda env create --file environment.yml
conda activate dynamic_gaussians

# Install rendering code (cuda)
git clone git@github.com:JonathonLuiten/diff-gaussian-rasterization-w-depth.git
cd diff-gaussian-rasterization-w-depth
python setup.py install
pip install .
```

## Run visualizer on pretrained models (original repo)
```bash
cd Dynamic3DGaussians
wget https://omnomnom.vision.rwth-aachen.de/data/Dynamic3DGaussians/output.zip  # Download pretrained models
unzip output.zip
python visualize.py  # See code for visualization options
```

## Train models yourself (original repo)
```bash
cd Dynamic3DGaussians
wget https://omnomnom.vision.rwth-aachen.de/data/Dynamic3DGaussians/data.zip  # Download training data
unzip data.zip
python train.py 
```

## Code Structure (original repo)
I tried really hard to make this code really clean and useful for building upon. In my opinion it is now much nicer than the original code it was built upon.
Everything is relatively 'functional' and I tried to remove redundant classes and modules wherever possible. 
Almost all of the code is in [train.py](./train.py) in a few core functions, with the overall training loop clearly laid out.
There are only a few other helper functions used, divided between [helpers.py](helpers.py) and [external.py](external.py) (depending on license).
I have split all useful variables into two dicts: 'params' (those updated with gradient descent), and 'variables' (those not updated by gradient descent).
There is also a custom visualization codebase build using Open3D (used for the cool visuals on the website) that is entirely in [visualize.py](visualize.py).
Please let me know if there is anyway you think the code could be cleaner. 

# Data making
## Blender: known camera poses and ground truth point cloud

This repository contains two important scripts to extract data from Blender:
- `data_making/blender_script.py`: **copy** this to Blender. This script extracts information about cameras in the scene + points on the object surface.
- `data_making/blender_to_data.py`: run this after extracting data from Blender to structure your data in the correct format.

Let's see how these two scripts work in detail.
### `blender_script.py`
- First, you need to create a scene in Blender:

<img alt='Blender scene showing a red toaster on a green plane' src="images/blender_scene.png" width="500">

- Copy the script to the text editor, set the parameters at the top of the script, and run it. This script generates N cameras uniformly placed on the surface of an hemisphere built around the object. All cameras point towards the object. **Important**: please set your paths to save the extracted data in the same folder, e.g.
```
output_img_path = 'PROJECT_FOLDER/ims'
output_poses_path = 'PROJECT_FOLDER/cameras_gt.json' 
output_point_path = 'PROJECT_FOLDER/init_pt_cld.npz' 
```

<img alt='Blender scene showing a red toaster on a green plane' src="images/cameras_blender.png" width="500">

### blender_to_data.py
Simply run the script from your root directory. Set the arguments as required, e.g
```
python blender_to_data.py --data_path Blender/PROJECT_FOLDER/ --output_path data/YOUR_DATASET
```

## Extract data using COLMAP
TODO.

## Notes on license (original repo)
The code in this repository (except in external.py) is licensed under the MIT licence.

However, for this code to run it uses the cuda rasterizer code from [here](https://github.com/JonathonLuiten/diff-gaussian-rasterization-w-depth),
as well as various code in [external.py](./external.py) which has been taken or adapted from [here](https://github.com/graphdeco-inria/gaussian-splatting).
These are required for this project, and for these a much more restrictive license from Inria applies which can be found [here](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md).
This requires express permission (licensing agreements) from Inria for use in any commercial application, but is otherwise freely distributed for research and experimentation.


## Citation
```
@inproceedings{luiten2023dynamic,
  title={Dynamic 3D Gaussians: Tracking by Persistent Dynamic View Synthesis},
  author={Luiten, Jonathon and Kopanas, Georgios and Leibe, Bastian and Ramanan, Deva},
  booktitle={3DV},
  year={2024}
}
```
