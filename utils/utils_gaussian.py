"""Script to extract mesh from Gaussian Splatting.
Code is adapted from https://github.com/dreamgaussian/dreamgaussian"""
import torch
from external import build_rotation

def gaussian_3d_coeff(xyzs, covs):
    """Compute Gaussians exponent"""
    # xyzs: [N, 3]
    # covs: [N, 6]
    x, y, z = xyzs[:, 0], xyzs[:, 1], xyzs[:, 2]
    a, b, c, d, e, f = covs[:, 0], covs[:, 1], covs[:, 2], covs[:, 3], covs[:, 4], covs[:, 5]

    # eps must be small enough !!!
    det = a * d * f + 2 * e * c * b - e**2 * a - c**2 * d - b**2 * f
    safe_det = torch.clamp(det, min=1e-10)
    inv_det = 1 / safe_det

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

def calculate_density(
        params,
        depth_pt_cld,
        variables,
        max_NN=10
    ):
    """Extract density for points in the point cloud. Compute occupancy by evaluating
    the contribution of all the Gaussians to each point. 
    Args:
        - params: dict containing Gaussian parameters, especially:
            - params['logit_opacities']: torch.tensor of opacities, [N, 1]
            - params['means3D']: torch.tensor of Gaussian means, [N, 3]
            - params['log_scales']: torch.tensor of scaling factors (std deviations), [N, 3]
            - params['unnorm_rotations']: torch.tensor of unnorm quaternions, [N, 4]

    Returns:
        occupancy values, [resolution, resolution, resolution]     
    """
    # Only consider depth gaussians
    indices_gaussians_depth = torch.nonzero(variables['depth'])[:, 0]
    logit_opacities = params['logit_opacities'][indices_gaussians_depth]
    means3D = params['means3D'][indices_gaussians_depth]
    log_scales = params['log_scales'][indices_gaussians_depth]
    unnorm_rotations = params['unnorm_rotations'][indices_gaussians_depth]

    covs = build_covariance_from_scaling_rotation(torch.exp(log_scales), 1, unnorm_rotations) # [N, 6]

    # Query per point-gaussian distances.
    g_pts = depth_pt_cld.unsqueeze(1) - means3D.unsqueeze(0) # [M, 1, 3] - [1, N, 3] = [M, N, 3]
    g_covs = covs.unsqueeze(0).repeat(depth_pt_cld.shape[0], 1, 1) # [M, N, 6]
    
    w = gaussian_3d_coeff(g_pts.reshape(-1, 3), g_covs.reshape(-1, 6)).reshape(depth_pt_cld.shape[0], -1) # [MxN] -> [M, N]
    
    if w.shape[1] > max_NN:

        top_values, top_indices = torch.topk(w, max_NN, dim=1)
        mask = torch.zeros_like(w).cuda()
        mask.scatter_(1, top_indices, 1)
        
        w = w * mask # [M, N]

    occ = w @ torch.sigmoid(logit_opacities) # [M, 1] 
    occ = occ.mean()

    return occ


def calculate_density_all(
        params,
        depth_pt_cld,
        variables
    ):
    """Extract density for points in the point cloud. Compute occupancy by evaluating
    the contribution of all the Gaussians to each point. 
    Args:
        - params: dict containing Gaussian parameters, especially:
            - params['logit_opacities']: torch.tensor of opacities, [N, 1]
            - params['means3D']: torch.tensor of Gaussian means, [N, 3]
            - params['log_scales']: torch.tensor of scaling factors (std deviations), [N, 3]
            - params['unnorm_rotations']: torch.tensor of unnorm quaternions, [N, 4]

    Returns:
        occupancy values, [resolution, resolution, resolution]     
    """
    covs = build_covariance_from_scaling_rotation(torch.exp(params['log_scales']), 1, params['unnorm_rotations']) # [N, 6]

    # Query per point-gaussian distances.
    g_pts = depth_pt_cld.unsqueeze(1) - params['means3D'].unsqueeze(0) # [M, 1, 3] - [1, N, 3] = [M, N, 3]
    g_covs = covs.unsqueeze(0).repeat(depth_pt_cld.shape[0], 1, 1) # [M, N, 6]

    w = gaussian_3d_coeff(g_pts.reshape(-1, 3), g_covs.reshape(-1, 6)).reshape(depth_pt_cld.shape[0], -1) # [MxN] -> [M, N]
    occ = w @ torch.sigmoid(params['logit_opacities']) # [M, 1] 
    occ = occ.mean()
    
    return occ


# Define a function that computes the value and returns its gradient vector
def calculate_gradient(params, depth_pt_cld, normals, variables, f):
    """Compute the gradient of the density function w.r.t. depth_pt_cld. Project the gradient onto the normals
    passing through the depth_pt_cld."""

    # With gradients ############################################
    depth_pt_cld.requires_grad_(True)
    value = f(params, depth_pt_cld, variables).mean()  # Average density over the point cloud

    # # Compute the first-order gradients (gradients of value w.r.t. depth pt cld), not w.r.t params
    grads = torch.autograd.grad(
        outputs=value,
        inputs=depth_pt_cld,
        create_graph=True) # [M, 3], directions for depth_pt_cld to follow to increase the value
    depth_pt_cld.requires_grad_(False)

    grads_proj = torch.sum(grads[0] * normals, dim=1).mean() # [M, 1], project the gradients onto the normals
    
    # Debug: check that the function computed outside is lower/higher than the one computed inside
    # It is higher inside for density while it is lower inside for transparency
    x1 = depth_pt_cld.clone() + 0.001 * normals.clone()
    x1.requires_grad_(True)
    x2 = depth_pt_cld.clone() - 0.001 * normals.clone()
    x2.requires_grad_(True)
    value1 = f(params, x1, variables)
    value2 = f(params, x2, variables)
    print('Function:', value1.item(), value.item(), value2.item())
    grads1 = torch.autograd.grad(
        outputs=value1,
        inputs=x1,
        create_graph=True) # [M, 3], directions for depth_pt_cld to follow to increase the value
    x1.requires_grad_(False)
    grads2 = torch.autograd.grad(
        outputs=value2,
        inputs=x2,
        create_graph=True) # [M, 3], directions for depth_pt_cld to follow to increase the value
    x2.requires_grad_(False)
    grads1_proj = torch.sum(grads1[0] * normals, dim=1).mean() # [M, 1], project the gradients onto the normals
    grads2_proj = torch.sum(grads2[0] * normals, dim=1).mean() # [M, 1], project the gradients onto the normals
    print('Grads:', grads1_proj.item(), grads_proj.item(), grads2_proj.item())

    return grads_proj


def calculate_transparency(
        params,
        depth_pt_cld,
        variables,
        max_NN=10
    ):
    """Compute the transparency of the gaussians as the product of 1 - opacity for all gaussians"""

    # Only consider depth gaussians
    indices_gaussians_depth = torch.nonzero(variables['depth'])[:, 0]
    logit_opacities = params['logit_opacities'][indices_gaussians_depth]
    means3D = params['means3D'][indices_gaussians_depth]
    log_scales = params['log_scales'][indices_gaussians_depth]
    unnorm_rotations = params['unnorm_rotations'][indices_gaussians_depth]

    covs = build_covariance_from_scaling_rotation(torch.exp(log_scales), 1, unnorm_rotations) # [N, 6]

    # Query per point-gaussian distances.
    g_pts = depth_pt_cld.unsqueeze(1) - means3D.unsqueeze(0) # [M, 1, 3] - [1, N, 3] = [M, N, 3]
    g_covs = covs.unsqueeze(0).repeat(depth_pt_cld.shape[0], 1, 1) # [M, N, 6]

    w = gaussian_3d_coeff(g_pts.reshape(-1, 3), g_covs.reshape(-1, 6)).reshape(depth_pt_cld.shape[0], -1) # [MxN] -> [M, N]
    
    if w.shape[1] > max_NN:

        _, top_indices = torch.topk(w, max_NN, dim=1)
        mask = torch.zeros_like(w).cuda()
        mask.scatter_(1, top_indices, 1)
        
        w = w * mask # [M, N]

    opacities_reshaped = torch.sigmoid(logit_opacities).T # [1, N]
    t = 1 - w * opacities_reshaped # [M, N] x [1, N] = [M, N] 
    T = torch.prod(t, dim=1) # [M, 1]

    T_mean = T.mean()
    
    return T_mean


# Define a function that computes the value and returns its gradient vector
def finite_element_transparency(params, depth_pt_cld, normals, variables, f, value_on):
    """Compute the gradient of the density function w.r.t. depth_pt_cld. Project the gradient onto the normals
    passing through the depth_pt_cld."""

    # With gradients ############################################
    if value_on is None:
        value_on = f(params, depth_pt_cld, variables)
    else:
        value_on = value_on.clone()
    x1 = depth_pt_cld + 0.01 * normals
    value_out = f(params, x1, variables) # Transparency over the point cloud + 0.001 * normals
    
    delta = value_out - value_on  # we want delta out-surface to be as high as possible
            
    return delta


# Define a function that computes the value and returns its gradient vector
def finite_element_transparency_fast(params,
        depth_pt_cld,
        variables,
        normals,
        max_NN=10
    ):
    """Compute the gradient of the density function w.r.t. depth_pt_cld. Project the gradient onto the normals
    passing through the depth_pt_cld."""

    """Compute the transparency of the gaussians as the product of 1 - opacity for all gaussians"""

    # Only consider depth gaussians
    indices_gaussians_depth = torch.nonzero(variables['depth'])[:, 0]
    logit_opacities = params['logit_opacities'][indices_gaussians_depth]
    means3D = params['means3D'][indices_gaussians_depth]
    log_scales = params['log_scales'][indices_gaussians_depth]
    unnorm_rotations = params['unnorm_rotations'][indices_gaussians_depth]

    covs = build_covariance_from_scaling_rotation(torch.exp(log_scales), 1, unnorm_rotations) # [N, 6]

    # Join all depth point clouds
    depth_pt_cld_out = depth_pt_cld + 0.01 * normals
    depth_pt_cld_all = torch.cat((depth_pt_cld, depth_pt_cld_out), dim=0) # [2M, 3]

    # Query per point-gaussian distances.
    g_pts = depth_pt_cld_all.unsqueeze(1) - means3D.unsqueeze(0) # [2M, 1, 3] - [1, N, 3] = [2M, N, 3]
    g_covs = covs.unsqueeze(0).repeat(depth_pt_cld_all.shape[0], 1, 1) # [2M, N, 6]

    w = gaussian_3d_coeff(g_pts.reshape(-1, 3), g_covs.reshape(-1, 6)).reshape(depth_pt_cld_all.shape[0], -1) # [2MxN] -> [2M, N]
    
    if w.shape[1] > max_NN:

        _, top_indices = torch.topk(w, max_NN, dim=1)
        mask = torch.zeros_like(w).cuda()
        mask.scatter_(1, top_indices, 1)
        
        w = w * mask # [2M, N]

    opacities_reshaped = torch.sigmoid(logit_opacities).T # [1, N]
    t = 1 - w * opacities_reshaped # [2M, N] x [1, N] = [2M, N] 
    t = t.reshape(2, -1, t.shape[1]) # [2, M, N]
    #T = torch.prod(t, dim=1) # [2, M, 1]
    T = torch.prod(t, dim=2) # [2, M, 1]
    T_mean = T.mean(dim=1) # [2, 1]
    delta_T = T_mean[1] - T_mean[0] # we want delta out-surface to be as high as possible
    return delta_T