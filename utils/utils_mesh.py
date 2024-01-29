import torch
import mcubes
from utils.utils_gaussian import gaussian_3d_coeff, build_covariance_from_scaling_rotation
import open3d as o3d
import plotly.graph_objects as go
import math
import time
import trimesh

# Code adapted from https://github.com/dreamgaussian/dreamgaussian
@torch.no_grad()
def extract_transmittance(
        opacities,
        xyzs,
        stds,
        unnorm_rotation,
        custom_mn,
        custom_mx,
        resolution, 
        num_blocks,
        relax_ratio
    ):
    """Extract occupancy from volumetric grid. Compute occupancy values in a 3D volume by evaluating
    the contribution of multiple Gaussians. The volume is divided into N voxels, which are further subdivided
    into smaller blocks.
    Args:
        - opacities: torch.tensor of opacities, [N, 1]
        - xyzs: torch.tensor of Gaussian means, [N, 3]
        - stds: torch.tensor of scaling factors (std deviations), [N, 3]
        - unnorm_rotation: torch.tensor of unnorm quaternions, [N, 4]
        - custom_mn: torch.tensor of minimum coordinates of the bounding box to extract, 
                     if None the bbox is computed automatically based on the Gaussians, [3,]
        - custom_mx: torch.tensor of maximum coordinates of the bounding box to extract, 
                     if None the bbox is computed automatically based on the Gaussians, [3,]
        - resolution: resolution of the grid over which to evaluate Gaussians, int
        - num_blocks: each volume of the bbox is further divided into M num_blocks, int
    Returns:
        occupancy values, [resolution, resolution, resolution]     
    """

    block_size = 2 / num_blocks

    assert resolution % block_size == 0
    split_size = resolution // num_blocks

    # normalize to ~ [-1, 1]
    if custom_mn != None and custom_mx!=None:
        mn, mx = custom_mn, custom_mx
    else:
        mn, mx = xyzs.amin(0), xyzs.amax(0)
        
    # Compute the scaling factor. The value 1.8 is chosen to scale this maximum range to fit within 
    # a specific normalized range: the target range is slightly less than [-1, 1] (which would be 2.0 in total).
    # This scaling leaves a small margin around the data in the normalized space.
    center = (mn + mx) / 2
    scale = 1.8 / (mx - mn).amax().item()
    
    # Normalise and apply scaling to the data
    xyzs = (xyzs - center) * scale
    stds = stds * scale
    
    covs = build_covariance_from_scaling_rotation(stds, 1, unnorm_rotation)

    # tile
    device = opacities.device
    occ = torch.zeros([resolution] * 3, dtype=torch.float32, device=device)

    X = torch.linspace(-1, 1, resolution, device=device).split(split_size)
    Y = torch.linspace(-1, 1, resolution, device=device).split(split_size)
    Z = torch.linspace(-1, 1, resolution, device=device).split(split_size)
    
    batch_g = 4096

    # loop blocks (assume max size of gaussian is smaller than relax_ratio * block_size !!!)
    for xi, xs in enumerate(X):
        for yi, ys in enumerate(Y):
            for zi, zs in enumerate(Z):
                xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing='ij')
                
                # Sample points [M, 3] within the current block
                pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).to(device)

                # In-tile gaussians mask
                vmin, vmax = pts.amin(0), pts.amax(0)
                # vmin and vmax are extended by (block_size * relax_ratio) to consider Gaussians 
                # that might affect points near the block's edges.
                vmin -= block_size * relax_ratio
                vmax += block_size * relax_ratio
                # Only retain gaussians that are within the block boundaries
                mask = (xyzs < vmax).all(-1) & (xyzs > vmin).all(-1)
                # If hit no gaussian, continue to next block
                if not mask.any():
                    continue
                mask_xyzs = xyzs[mask] # [L, 3]
                mask_covs = covs[mask] # [L, 6]
                mask_opas = opacities[mask].view(1, -1) # [L, 1] --> [1, L]

                g_pts = pts.unsqueeze(1).repeat(1, mask_covs.shape[0], 1) - mask_xyzs.unsqueeze(0) # [M, L, 3]
                g_covs = mask_covs.unsqueeze(0).repeat(pts.shape[0], 1, 1) # [M, L, 6]

                # Batch on gaussian to avoid OOM.Computes the contribution of the gaussians withing the block
                # to the query point as weighted average of occupancies
                val = 0
                for start in range(0, g_covs.shape[1], batch_g):
                    end = min(start + batch_g, g_covs.shape[1])
                    # Contribution of gaussians at each query point 
                    w = gaussian_3d_coeff(g_pts[:, start:end].reshape(-1, 3), g_covs[:, start:end].reshape(-1, 6)).reshape(pts.shape[0], -1) # [M, l]
                    # val += torch.prod(1 - mask_opas[:, start:end] * w, dim=-1)
                    val += torch.amin(1 - mask_opas[:, start:end] * w, dim=-1)


                occ[xi * split_size: xi * split_size + len(xs), 
                    yi * split_size: yi * split_size + len(ys), 
                    zi * split_size: zi * split_size + len(zs)] = val.reshape(len(xs), len(ys), len(zs)) 
    
    return occ



# Code adapted from https://github.com/dreamgaussian/dreamgaussian
@torch.no_grad()
def extract_fields(
        opacities,
        xyzs,
        stds,
        unnorm_rotation,
        custom_mn,
        custom_mx,
        resolution=128, 
        num_blocks=32,
        relax_ratio=1.5
    ):
    """Extract occupancy from volumetric grid. Compute occupancy values in a 3D volume by evaluating
    the contribution of multiple Gaussians. The volume is divided into N voxels, which are further subdivided
    into smaller blocks.
    Args:
        - opacities: torch.tensor of opacities, [N, 1]
        - xyzs: torch.tensor of Gaussian means, [N, 3]
        - stds: torch.tensor of scaling factors (std deviations), [N, 3]
        - unnorm_rotation: torch.tensor of unnorm quaternions, [N, 4]
        - custom_mn: torch.tensor of minimum coordinates of the bounding box to extract, 
                     if None the bbox is computed automatically based on the Gaussians, [3,]
        - custom_mx: torch.tensor of maximum coordinates of the bounding box to extract, 
                     if None the bbox is computed automatically based on the Gaussians, [3,]
        - resolution: resolution of the grid over which to evaluate Gaussians, int
        - num_blocks: each volume of the bbox is further divided into M num_blocks, int
    Returns:
        occupancy values, [resolution, resolution, resolution]     
    """

    block_size = 2 / num_blocks

    assert resolution % block_size == 0
    split_size = resolution // num_blocks

    # normalize to ~ [-1, 1]
    if custom_mn != None and custom_mx!=None:
        mn, mx = custom_mn, custom_mx
    else:
        mn, mx = xyzs.amin(0), xyzs.amax(0)
        
    # Compute the scaling factor. The value 1.8 is chosen to scale this maximum range to fit within 
    # a specific normalized range: the target range is slightly less than [-1, 1] (which would be 2.0 in total).
    # This scaling leaves a small margin around the data in the normalized space.
    center = (mn + mx) / 2
    scale = 1.8 / (mx - mn).amax().item()
    
    # Apply scaling to the data
    xyzs = (xyzs - center) * scale
    stds = stds * scale
    
    covs = build_covariance_from_scaling_rotation(stds, 1, unnorm_rotation)

    # tile
    device = opacities.device
    occ = torch.zeros([resolution] * 3, dtype=torch.float32, device=device)

    X = torch.linspace(-1, 1, resolution).split(split_size)
    Y = torch.linspace(-1, 1, resolution).split(split_size)
    Z = torch.linspace(-1, 1, resolution).split(split_size)


    # loop blocks (assume max size of gaussian is small than relax_ratio * block_size !!!)
    for xi, xs in enumerate(X):
        for yi, ys in enumerate(Y):
            for zi, zs in enumerate(Z):
                xx, yy, zz = torch.meshgrid(xs, ys, zs)
                
                # Sample points [M, 3] within the current block
                pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).to(device)
                
                # In-tile gaussians mask
                vmin, vmax = pts.amin(0), pts.amax(0)
                # vmin and vmax are extended by (block_size * relax_ratio) to consider Gaussians 
                # that might affect points near the block's edges.
                vmin -= block_size * relax_ratio
                vmax += block_size * relax_ratio
                # Only retain gaussians that are within the block boundaries
                mask = (xyzs < vmax).all(-1) & (xyzs > vmin).all(-1)
                # If hit no gaussian, continue to next block
                if not mask.any():
                    continue
                mask_xyzs = xyzs[mask] # [L, 3]
                mask_covs = covs[mask] # [L, 6]
                mask_opas = opacities[mask].view(1, -1) # [L, 1] --> [1, L]

                # Query per point-gaussian distances.
                g_pts = pts.unsqueeze(1).repeat(1, mask_covs.shape[0], 1) - mask_xyzs.unsqueeze(0) # [M, L, 3]
                g_covs = mask_covs.unsqueeze(0).repeat(pts.shape[0], 1, 1) # [M, L, 6]

                # Batch on gaussian to avoid OOM.Computes the contribution of the gaussians withing the block
                # to the query point as weighted average of occupancies
                batch_g = 1024
                val = 0
                for start in range(0, g_covs.shape[1], batch_g):
                    end = min(start + batch_g, g_covs.shape[1])
                    # Contribution of gaussians at each query point 
                    w = gaussian_3d_coeff(g_pts[:, start:end].reshape(-1, 3), g_covs[:, start:end].reshape(-1, 6)).reshape(pts.shape[0], -1) # [M, l]
                    val += (mask_opas[:, start:end] * w).sum(-1)

                occ[xi * split_size: xi * split_size + len(xs), 
                    yi * split_size: yi * split_size + len(ys), 
                    zi * split_size: zi * split_size + len(zs)] = val.reshape(len(xs), len(ys), len(zs)) 
    
    return occ



# Code adapted from https://github.com/dreamgaussian/dreamgaussian
def extract_mesh(
        opacities,
        xyzs,
        stds,
        unnorm_rotations,
        custom_mn=None,
        custom_mx=None,
        density_thresh=0.5,
        resolution=128,
        num_blocks=16,
        relax_ratio=1.5,
        decimate_target=1e5
    ):
    """Method to extract mesh from Gaussian Splatting."""

    start = time.time()
    occ = extract_transmittance(
        opacities,
        xyzs,
        stds,
        unnorm_rotations,
        custom_mn,
        custom_mx,
        resolution,
        num_blocks,
        relax_ratio
    ).detach().cpu().numpy()
    print(f'Extract transmittance: {time.time() - start}')
    print(occ.mean())

    start = time.time()
    vertices, triangles = mcubes.marching_cubes(occ, density_thresh)
    print(f'Extract mesh: {time.time() - start}')
    vertices = vertices / (resolution - 1.0) * 2 - 1
    
    return vertices, triangles


def fibonacci_hemisphere(samples, sphere_radius):
    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # Golden angle in radians

    for i in range(samples):
        z = (i / float(samples - 1))  # Range from 0 to 1
        radius = math.sqrt(1 - z * z)  # Radius at y

        theta = phi * i  # Increment

        x = math.cos(theta) * radius * sphere_radius
        y = math.sin(theta) * radius * sphere_radius

        points.append((x, y, z * sphere_radius))

    return points

def estimate_normals(depth_pt_cld):
    """Given a point cloud, estimate normals.
    Params:
        - depth_pt_cld: [N, 3] torch.tensor"""
    
    # To align the normals correctly (ouward), we sample random points on a virtual
    # hemisphere surrounding the object. We compute the closest distance to a point. 
    # the position of that point determines the direction of the normal. 
    points_on_hemisphere = fibonacci_hemisphere(1000, 10)
    points_on_hemisphere = torch.tensor(points_on_hemisphere, dtype=torch.float32).cuda() # [1000, 3]
    diff = points_on_hemisphere.unsqueeze(0) - depth_pt_cld.unsqueeze(1)   # [1, 1000, 3] - [N, 1, 3] --> [N, 1000, 3]
    dist = torch.norm(diff, dim=-1) # [N, 1000]
    idx_dist_min = dist.argmin(dim=-1) # [N]
    dist_min = diff[torch.arange(dist.shape[0]), idx_dist_min]

    # Compute normals
    point_cloud_o3d = o3d.geometry.PointCloud()
    point_cloud_o3d.points = o3d.utility.Vector3dVector(depth_pt_cld.cpu().numpy())
    point_cloud_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    normals = torch.tensor(point_cloud_o3d.normals, dtype=torch.float32).cuda()

    dot_prod = torch.sum(normals * dist_min, dim=-1)

    # Flip normals if they are not pointing towards the camera
    normals[dot_prod < 0] *= -1


    return normals


def _debug_normals(depth_pt_cld, normals):
    # Plot normals
    fig = go.Figure(data=[go.Scatter3d(
        x=depth_pt_cld.cpu()[:, 0],
        y=depth_pt_cld.cpu()[:, 1],
        z=depth_pt_cld.cpu()[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color='red',
            opacity=0.8
        )
    )])
    fig.add_trace(go.Cone(
        x=depth_pt_cld.cpu()[:, 0],
        y=depth_pt_cld.cpu()[:, 1],
        z=depth_pt_cld.cpu()[:, 2],
        u=normals.cpu()[:, 0],
        v=normals.cpu()[:, 1],
        w=normals.cpu()[:, 2],
        sizemode="absolute",
        sizeref=1,
        anchor="tail"
    ))
    fig.show()

    # Plot depth points and displaced points
    disp_points = depth_pt_cld + 0.1 * normals
    fig = go.Figure(data=[go.Scatter3d(
        x=depth_pt_cld.cpu()[:, 0],
        y=depth_pt_cld.cpu()[:, 1],
        z=depth_pt_cld.cpu()[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color='red',
            opacity=0.8
        )
    )])
    fig.add_trace(go.Scatter3d(
        x=disp_points.cpu()[:, 0],
        y=disp_points.cpu()[:, 1],
        z=disp_points.cpu()[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color='green',
            opacity=0.8
        )
    ))
    fig.show()

def as_mesh(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([
            trimesh.Trimesh(vertices=m.vertices, faces=m.faces)
            for m in scene_or_mesh.geometry.values()])
    else:
        mesh = scene_or_mesh
    return mesh
