import torch
from utils.utils_mesh import extract_mesh
import argparse
import os
import output
import numpy as np
import trimesh

def load_output(output_name):
    path = os.path.join(os.path.dirname(output.__file__), f'{output_name}/params.npz')
    output_dict = np.load(path)

    return output_dict
    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_name", default='', type=str, help="Name of the experiment/run, e.g. 'exp1/toaster_refl'"
    )
    args = parser.parse_args()

    args.output_name = 'exp1/toaster_refl_transparency_FE'

    output_dict = load_output(args.output_name)
    
    # Define the same variables as above and extract mesh with constraints on the domain
    opacities = torch.sigmoid(torch.tensor(output_dict['logit_opacities'])).cuda()
    xyzs = torch.tensor(output_dict["means3D"])[0].cuda()
    stds = torch.exp(torch.tensor(output_dict['log_scales'])).cuda()
    unnorm_rotations = torch.tensor(output_dict['unnorm_rotations'])[0].cuda()
    custom_mn = torch.tensor([-1, -1, -0.5]).cuda()
    custom_mx = torch.tensor([1, 1, 0.5]).cuda() 
    vertices, triangles = extract_mesh(
        opacities,
        xyzs, stds,
        unnorm_rotations,
        custom_mn=custom_mn, custom_mx=custom_mx,
        density_thresh=0.05,
        resolution=128,
        num_blocks=8,
        relax_ratio=1.5
    )

    mesh = trimesh.Trimesh(vertices, triangles)
    mesh.show()
