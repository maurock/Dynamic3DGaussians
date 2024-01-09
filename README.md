# Depth-Aware Dynamic 3D Gaussians
This repository extends [Dynamic 3D Gaussians:
Tracking by Persistent Dynamic View Synthesis](https://dynamic3dgaussians.github.io/) to reconstruct challenging objects, including reflective and transparent materials. We introduce a method for incorporating explicit depth hints and evaluate it against state-of-the-art approaches. <br><br>

Key additions to this repository include:
- Code and intructions for data extraction using Blender. [Go to the instructions](#)
- An upgraded Visualiser tool featuring interactive navigation through known camera poses, mode switching ('colors', 'depth'), and other enhancements.
- Scripts for converting various datasets to the necessary format: [3DGS], [RefNeRF](https://dorverbin.github.io/refnerf/)
- Minor changes and comments for clarity.

Please consider citing our work and the original implementation if you find this repository helpful:

```
@inproceedings{luiten2023dynamic,
  title={Depth-Aware Dynamic 3D Gaussians},
  author={Comi, Mauro and Tonioni, Alessio and Aitchison, Laurence and Lepora, Natahn F.},
  year={2024},
  publisher={GitHub},
  journal={GitHub repository},
  howpublished={\url{https://github.com/maurock/Dynamic3DGaussians}},
}
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
## Blender: accurate camera poses and ground truth point cloud

This repository contains two important scripts for data extraction from Blender:
- `data_making/blender_script.py`: **Copy** this to your Blender script. This script gathers details about the cameras in your 3D scene and points on the surface of your object(s).  
- `data_making/blender_to_data.py`: Run this script post data extraction from Blender to structure your data in the correct format.

Below is an explanation of how these scripts work:
### `blender_script.py`
- Start by creating your object-centric scene in Blender:

<img alt='Blender scene showing a red toaster on a green plane' src="images/blender_scene.png" width="500">

- Paste the script into Blender's text editor, set the parameters at the beginning of the script, and run it. This script generates N cameras, uniformly distributed on a hemisphere surrounding the object, with each camera oriented towards the object. **Important**: please ensure that your file paths to save the extracted data are in a unified directory, e.g.
```
output_img_path = 'PROJECT_FOLDER/ims'
output_poses_path = 'PROJECT_FOLDER/cameras_gt.json' 
output_point_path = 'PROJECT_FOLDER/init_pt_cld.npz' 
```

<img alt='Blender scene showing a red toaster on a green plane' src="images/cameras_blender.png" width="500">

### blender_to_data.py
Simply run this script from your root directory. Configure the arguments as needed, e.g
```
python blender_to_data.py --data_path Blender/PROJECT_FOLDER/ --output_path data/YOUR_DATASET
```

## Extract data using COLMAP
Coming soon.

## RefNeRF
The script .`data_making/refnerf_to_data.py` is included for converting the [RefNeRF](https://dorverbin.github.io/refnerf/) dataset, which contains data for shiny and reflective objects, to the format required by this repository. 
Please follow these steps:
- Download the **Shiny Dataset** from the official [RefNeRF project page](https://dorverbin.github.io/refnerf/).
- Place the downloaded folder (`refnerf`) in the `data/` folder.
- Run `python data_making/refnerf_to_data.py` to convert the data in the required format. This script also runs COLMAP with the known camera poses included in the RefNeRF. The converted data is stored in `data/`, e.g. `data/toaster`, ready to be used by the `train.py` script. Please note: local depth information is not extracted. Please follow the next steps for it.
- **ADD INSTRUCTIONS TO EXTRACT DEPTH**


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
