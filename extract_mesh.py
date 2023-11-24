"""Script to extract mesh from Gaussian Splatting.
Code is adapted from https://github.com/dreamgaussian/dreamgaussian"""
import torch
import mcubes
from external import build_rotation

def gaussian_3d_coeff(xyzs, covs):
    """Compute Gaussians exponent"""
    # xyzs: [N, 3]
    # covs: [N, 6]
    x, y, z = xyzs[:, 0], xyzs[:, 1], xyzs[:, 2]
    a, b, c, d, e, f = covs[:, 0], covs[:, 1], covs[:, 2], covs[:, 3], covs[:, 4], covs[:, 5]

    # eps must be small enough !!!
    inv_det = 1 / (a * d * f + 2 * e * c * b - e**2 * a - c**2 * d - b**2 * f + 1e-24)
    inv_a = (d * f - e**2) * inv_det
    inv_b = (e * c - b * f) * inv_det
    inv_c = (e * b - c * d) * inv_det
    inv_d = (a * f - c**2) * inv_det
    inv_e = (b * c - e * a) * inv_det
    inv_f = (a * d - b**2) * inv_det

    power = -0.5 * (x**2 * inv_a + y**2 * inv_d + z**2 * inv_f) - x * y * inv_b - x * z * inv_c - y * z * inv_e

    power[power > 0] = -1e10 # abnormal values... make weights 0
        
    return torch.exp(power)


def build_scaling_rotation(s, r):
    """Batch of 3D transformation matrices, each of which applies a rotation followed by a scaling. 
    Args:
        - r: unnormalised quaternions
        - s: scaling factors"""
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device='cuda:0')
    R = build_rotation(r)
    
    # Populate diagonal of L matrix with scaling factors
    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
    """Build covariance matrix and return lower diagonal"""
    L = build_scaling_rotation(scaling_modifier * scaling, rotation) # [N, 3, 3]
    # Multiply by its transpose: spectral theorem
    actual_covariance = L @ L.transpose(1, 2) # [N, 3, 3]
    # Store lower diaognal due to symmetries in the covariance matrix
    symm = strip_lowerdiag(actual_covariance) # [N, 6]
    return symm
        
def strip_lowerdiag(L):
    """Extract lower diagonal from covariance matrix.
    Args:
        - L: covariance matrix, [N, 3, 3]
    Return:
        - uncertainty: lower diagonal, [N, 6]
    """
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device='cuda:0')

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

@torch.no_grad()
def extract_fields(
        opacities,
        xyzs,
        stds,
        unnorm_rotation,
        custom_mn,
        custom_mx,
        resolution=128, 
        num_blocks=16,
        relax_ratio=1.5
    ):
    """Compute occupancy values in a 3D volume by evaluating the contribution of multiple Gaussians.
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

def extract_mesh(
        opacities,
        xyzs,
        stds,
        unnorm_rotations,
        custom_mn=None,
        custom_mx=None,
        density_thresh=1,
        resolution=128,
        decimate_target=1e5
    ):
    """Method to extract mesh from Gaussian Splatting."""
    occ = extract_fields(opacities, xyzs, stds, unnorm_rotations, custom_mn, custom_mx, resolution).detach().cpu().numpy()

    vertices, triangles = mcubes.marching_cubes(occ, density_thresh)
    vertices = vertices / (resolution - 1.0) * 2 - 1
    
    return vertices, triangles


if __name__=='__main__':
    pass