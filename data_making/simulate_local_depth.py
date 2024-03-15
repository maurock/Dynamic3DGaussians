"""
Simulate touches on a mesh from the RefNeRF dataset. 
The point clouds are stored in the data folder.

Author: Mauro Comi, mauro.comi@bristol.ac.uk
Date: 28/07/2021
"""
import numpy as np
import utils.utils_mesh as utils_mesh
import plotly.graph_objects as go
import trimesh
import argparse
import data
import os
from glob import glob
from utils import utils_data

def load_mesh(filename):
    scene_or_mesh = trimesh.load_mesh(filename)
    mesh = utils_mesh.as_mesh(scene_or_mesh)
    return mesh


def get_random_directions(n_dirs):
    '''Get random directions on the unit sphere'''
    # ShinyBlender
    # theta = np.random.uniform(0, 2 * np.pi / 3, size=(n_dirs, 1))  # Polar angle between 0 and π radians
    # phi = np.random.uniform(0, 2 * np.pi, size=(n_dirs, 1))  # Azimuthal angle between 0 and 2π radians

    # GlossySynthetic
    theta = np.random.uniform(0, np.pi, size=(n_dirs, 1))  # Polar angle between 0 and π radians
    phi = np.random.uniform(0, 2 * np.pi, size=(n_dirs, 1))  # Azimuthal angle between 0 and 2π radians

    x = 3 * np.sin(theta) * np.cos(phi)
    y = 3 * np.sin(theta) * np.sin(phi)
    z = 3 * np.cos(theta)
    directions = np.concatenate((x, y, z), axis=1)
    return directions


def get_intersections(mesh, center, num_rays):
    '''Cast rays radially from the centre of the object, get the furthest intersection per ray '''
    ray_origins = np.tile(center, (num_rays, 1))
    ray_directions = get_random_directions(num_rays)
    
    locations, index_ray, _ = mesh.ray.intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_directions
    )
    
    return locations, index_ray
        

def get_closest_points(pc, intersections, num_points_per_intersection):
    '''
    Get points from pc that are closest to intersections.
    Params:
        pc: np.array, shape (M, 3)
        intersections: np.array, shape (N, 3)
    '''
    points_touch = []
    dist = pc[None,:,:] - intersections[:,None,:]   # dist: (N, M, 3)
    dist = np.linalg.norm(dist, axis=2)  # dist: (N, M)

    # Get the indices of the closest points    
    closest_points_indices = np.array([np.argsort(x)[:num_points_per_intersection] for x in dist])

    points_touch = pc[closest_points_indices] 
    points_touch = points_touch.reshape(-1, 3)
    
    return points_touch


def debug_plot(points, intersections, depth_points=None):
    fig = go.Figure(data=[
        go.Scatter3d(
            x=points[:,0],
            y=points[:,1],
            z=points[:,2],
            mode='markers',
            marker=dict(
                size=2
            )
        ),
        go.Scatter3d(
            x=intersections[:,0],
            y=intersections[:,1],
            z=intersections[:,2],
            mode='markers',
            marker=dict(
                size=4
            )
        )
    ])
    if depth_points is not None:
        # Add Scatter3D for debug_points
        fig.add_trace(
            go.Scatter3d(
                x=depth_points[:, 0],
                y=depth_points[:, 1],
                z=depth_points[:, 2],
                mode='markers',
                marker=dict(
                    size=3,
                    color='cyan',  # Different color for debug points
                ),
                name='Debug Points'
            )
        )
    fig.show()


def main(args):
    # List all objects paths
    obj_paths = utils_data.get_obj_paths(args.dataset)

    if args.dataset == 'ShinyBlender':
        dataset_dir = 'shiny-blender-3DGS'
        pc_num = 10000
    elif args.dataset == 'GlossySynthetic':
        dataset_dir = 'glossy-synthetic-3DGS'
        pc_num = 20000
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}. Choose between 'ShinyBlender' and 'GlossySynthetic'.")

    for obj in obj_paths:
        # List of depth points
        depths_list = []
        
        # Load mesh
        mesh = load_mesh(obj)

        if dataset_dir == 'glossy-synthetic-3DGS':
            # rotate object by -90 degrees around x-axis
            mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0]))

        center = mesh.centroid
        
        # Points on the object surface
        pc = np.array(trimesh.sample.sample_surface_even(mesh, pc_num)[0])
        
        # Get all intersections
        for num_touches in range(1, args.max_num_touches):
            
            intersections = np.array([])
            while intersections.shape[0] == 0:
                intersections, index_rays = get_intersections(mesh, center, num_rays=1)
                
            # Get furthest intersections
            unique_index_rays = np.unique(index_rays)
            unique_intersections = []
            for unique_index_ray in unique_index_rays: 
                intersections_per_ray = intersections[index_rays == unique_index_ray]
                distances = np.linalg.norm(intersections_per_ray - center, axis=1)
                index = np.argmax(distances)
                unique_intersections.append(intersections_per_ray[index])
            unique_intersections = np.stack(unique_intersections)
           
            # Get points closest to the intersections
            depth_points  = get_closest_points(pc, unique_intersections, args.num_points_per_intersection)
            depths_list.append(depth_points)

        if dataset_dir == 'glossy-synthetic-3DGS':
            # rotate object by 90 degrees around x-axis
            Rx = trimesh.transformations.rotation_matrix(-np.pi/2, [1, 0, 0])[:3, :3]
            depths_list = Rx[None,...] @ np.array(depths_list).transpose(0,2,1)  # [1, 3, 3] @ [T, 3, P]
            depths_list = depths_list.transpose(0,2,1)  # [T, P, 3]

        obj_name = os.path.basename(obj).replace('.obj', '')
        if obj_name == 'musclecar':    # Fix mismatch between refnerf and refnerf-blend
            obj_name = 'car'
        output_path = os.path.join(
            os.path.dirname(data.__file__), dataset_dir, obj_name, f'depth_pt_cld.npz'
        )
        np.savez_compressed(output_path, depth_points=depths_list)
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='', help="Dataset name: choose between 'ShinyBlender' or 'GlossySynthetic'")
    parser.add_argument('--max_num_touches', type=int, default=500)
    parser.add_argument('--num_points_per_intersection', type=int, default=50)
    args = parser.parse_args()
    
    main(args)
