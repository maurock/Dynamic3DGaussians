import torch
import os
import json
import copy
import numpy as np
from PIL import Image
from random import randint
from tqdm import tqdm
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from helpers import setup_camera, l1_loss_v1, l1_loss_v2, weighted_l2_loss_v1, weighted_l2_loss_v2, quat_mult, \
    params2rendervar, params2cpu, save_params, save_variables, save_eval_helper, read_config, save_config, scipy_knn
from external import calc_ssim, calc_psnr, build_rotation, densify, update_params_and_optimizer, inverse_sigmoid, means_scheduler_args
from pytorch3d.loss import chamfer_distance
import argparse
from utils import utils_mesh, utils_gaussian, utils_data, utils_sh
import time
import os
import extract_output_data
import config
import data
from eval import Evaluator
import output
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# torch.autograd.set_detect_anomaly(True)

def initialise_depth_gaussians(seq, md, num_touches, random_selection, dataset):
    """
    Generate gaussians for depth point cloud. Differently from standard Gaussians,
    the 'depth' variable is set to 1.
    Returns:
        params: dict of parameters
        variables: dict of variables
        depth_pt_cld: torch tensor of depth point cloud, shape [N, 3]
    """  
    try:
        depth_pt_cld = np.load(f"{os.path.dirname(data.__file__)}/{seq}/depth_pt_cld.npz")['depth_points']
    except:
        print(f"Depth point cloud not found. Exiting.")
        exit()
        
    if random_selection:
        # Sample random touches
        indexes = np.random.choice(depth_pt_cld.shape[0], size=num_touches, replace=False)
        depth_pt_cld = depth_pt_cld[indexes].reshape(-1, 3)
    else:
        depth_pt_cld = depth_pt_cld[:num_touches].reshape(-1, 3)

    # seg set to 1 for depth gaussians
    seg = np.ones(shape=(depth_pt_cld.shape[0])) # segmented, e.g. [0, 0, 1, 1, 1 ..]
    # depth 1 for depth gaussians
    depth = torch.ones(size=(depth_pt_cld.shape[0],)).cuda().float()

    # Initialise SH
    shs_degree = 3
    shs_num_coeffs = (shs_degree + 1) ** 2
    shs_rest = np.zeros((depth_pt_cld.shape[0], shs_num_coeffs - 1, 3))
    shs_colors_dc = np.random.random((depth_pt_cld.shape[0], 3)) / 255.0

    # sq_dist_1, _ = o3d_knn(depth_pt_cld[:, :3], 3)
    dist, _ = scipy_knn(depth_pt_cld[:, :3], 3)
    sq_dist = (dist**2)
    mean3_sq_dist = sq_dist.mean(-1).clip(min=0.0000001, max=1)
    # params are updated with gradient descent
    params = {
        'means3D': depth_pt_cld,    # (num_gaussians, 3)
        #'rgb_colors': rgb,    # (num_gaussians, 3)
        'seg_colors': np.stack((seg, np.zeros_like(seg), 1 - seg), -1),
        'unnorm_rotations': np.tile([1, 0, 0, 0], (seg.shape[0], 1)),
        'logit_opacities': np.array(inverse_sigmoid(torch.ones(size=(depth_pt_cld.shape[0], 1)) * 0.99999)),
        'log_scales': np.tile(np.log(np.sqrt(mean3_sq_dist))[..., None], (1, 3)),
        'shs_dc': np.expand_dims(shs_colors_dc, 1),
        'shs_rest': shs_rest
    }
    params = {k: torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True)) for k, v in params.items()}

    cam_centers = np.linalg.inv(md['w2c'][0])[:, :3, 3]  # Get scene radius
    scene_radius = 1.1 * np.max(np.linalg.norm(cam_centers - np.mean(cam_centers, 0)[None], axis=-1))
    # variables are NOT updated with gradient descent
    variables = {'max_2D_radius': torch.zeros((params['means3D'].shape[0])).cuda().float(),
                 'scene_radius': scene_radius,
                 'means2D_gradient_accum': torch.zeros((params['means3D'].shape[0], 1)).cuda().float(),
                 'denom': torch.zeros((params['means3D'].shape[0], 1)).cuda().float(),
                 'depth': depth
                 }

    depth_pt_cld = torch.tensor(depth_pt_cld).cuda().float()
    
    return params, variables, depth_pt_cld


def combine_params_and_variables(params, params_depth, variables, variables_depth):
    """
    Combine the parameters and variables of the depth gaussians with the parameters and variables of the
    normal gaussians.
    """
    params_combined = {}
    variables_combined = {}
    with torch.no_grad():
        for key in params.keys():
            if key in params_depth.keys():
                params_combined[key] = torch.cat((params[key], params_depth[key]), dim=0)
            else:
                print(f"Key {key} not in params_depth.keys()")
                params_combined[key] = params[key]
        for key in variables.keys():
            if key in variables_depth.keys() and key != "scene_radius":
                variables_combined[key] = torch.cat((variables[key], variables_depth[key]), dim=0)
            elif key == "scene_radius":
                variables_combined[key] = variables[key]
            else:
                print(f"Key {key} not in params_depth.keys()")
                variables_combined[key] = variables[key]

    params_combined = {k: torch.nn.Parameter(v.clone().detach().cuda().float().contiguous().requires_grad_(True)) for k, v in
            params_combined.items()}
    
    return params_combined, variables_combined


def get_dataset(t, md, configs):
    """
    A dataset is a list of dictionaries, every element in the list refers to a camera at current current timestep `t`. 
    Example: 50 cameras -> `len(dataset) = 50`
    The keys are: `dict_keys(['cam', 'im', 'seg', 'id'])`` 
    - 'cam': obj of type GaussianRasterisationSetting
    - 'im': torch.tensor, shape [3, height, width] 
    - 'seg': torch.tensor, shape [3, height, width] 
    - 'id': int of current camera id
    """
    dataset = []
    seq = configs['input_seq']
    ratio = configs['ratio_data']

    data_indexes = utils_data.filter_dataset(len(md['fn'][t]), ratio)

    for c in data_indexes:  # md['fn'][t] is a list of the image paths at time t, e.g. [0/a.jpg, 3/a.jpg, ..] 
                                       # meaning camera 0 -> image a.jpg, camera 3 -> image a.jpg, etc.
        w, h, k, w2c = md['w'], md['h'], md['k'][t][c], md['w2c'][t][c]
        cam = setup_camera(w, h, k, w2c, near=0.01, far=100.0)
        fn = md['fn'][t][c]
        
        # IMPORTANT: load data with alpha blending (background and alpha channel)
        bg = cam.bg.cpu().numpy()  
        rgb_path = f"{os.path.dirname(data.__file__)}/{seq}/ims/{fn}"
        im = utils_data.load_single_rgb_gt(rgb_path, bg=bg).permute(2, 0, 1)

        seg = np.array(copy.deepcopy(Image.open(f"{os.path.dirname(data.__file__)}/{seq}/seg/{fn.replace('.jpg', '.png')}"))).astype(np.float32)
        seg = torch.tensor(seg).float().cuda()
        seg_col = torch.stack((seg, torch.zeros_like(seg), 1 - seg))

        dataset.append({'cam': cam, 'im': im, 'seg': seg_col, 'id': c})
    return dataset


def get_batch(todo_dataset, dataset):
    if not todo_dataset:
        todo_dataset = dataset.copy()
    curr_data = todo_dataset.pop(randint(0, len(todo_dataset) - 1))
    return curr_data


def initialize_params(seq, md):
    """
    Params:
        seq: name of the data sequence, e.g. "basketball"
        md: metadata
    """

    colmap_pt_cld = f"{os.path.dirname(data.__file__)}/{seq}/init_pt_cld.npz"

    if not os.path.exists(colmap_pt_cld):
        # Random initialisation of pointcloud
        init_pt_cld = np.random.random((100000, 3)) * 2.6 - 1.3
        shs_colors_dc = np.random.random((100000, 3)) / 255.0
        seg = np.ones(100000)
    else:
        # Initialisation of pointcloud based on COLMAP
        init_pt_cld = np.load(f"{os.path.dirname(data.__file__)}/{seq}/init_pt_cld.npz")["data"]
        seg = init_pt_cld[:, 6]   # segmented, e.g. [0, 0, 1, 1, 1 ..]
        shs_colors_dc = init_pt_cld[:, 3:6] / 255.   # colours

    # initialise SH
    shs_degree = 3
    shs_num_coeffs = (shs_degree + 1) ** 2
    shs_rest = np.zeros((init_pt_cld.shape[0], shs_num_coeffs - 1, 3)) 

    dist, _ = scipy_knn(init_pt_cld[:, :3], 3)   # return sq_sit of 3 closest points
    sq_dist = (dist**2)
    mean3_sq_dist = sq_dist.mean(-1).clip(min=0.0000001, max=1)
    # depth 0 for depth gaussians
    depth = torch.zeros(size=(init_pt_cld.shape[0],)).cuda().float()

    # params are updated with gradient descent
    params = {
        'means3D': init_pt_cld[:, :3],    # (num_gaussians, 3)
        #'rgb_colors': init_pt_cld[:, 3:6],    # (num_gaussians, 3)
        'seg_colors': np.stack((seg, np.zeros_like(seg), 1 - seg), -1),
        'unnorm_rotations': np.tile([1, 0, 0, 0], (seg.shape[0], 1)),
        'logit_opacities': np.array(inverse_sigmoid(0.1 * torch.ones((seg.shape[0], 1), dtype=torch.float))),
        'log_scales': np.tile(np.log(np.sqrt(mean3_sq_dist))[..., None], (1, 3)),
        'shs_dc': np.expand_dims(shs_colors_dc, 1),  # (num_gaussians, 1, 3)
        'shs_rest': shs_rest                         # (num_gaussians, shs_num_coeffs-1, 3)
    }
    params = {k: torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True)) for k, v in
              params.items()}
    
    cam_centers = np.linalg.inv(md['w2c'][0])[:, :3, 3]  # Get scene radius
    scene_radius = 1.1 * np.max(np.linalg.norm(cam_centers - np.mean(cam_centers, 0)[None], axis=-1))
    # variables are NOT updated with gradient descent
    variables = {'max_2D_radius': torch.zeros((params['means3D'].shape[0])).cuda().float(),
                 'scene_radius': scene_radius,
                 'means2D_gradient_accum': torch.zeros((params['means3D'].shape[0], 1)).cuda().float(),
                 'denom': torch.zeros((params['means3D'].shape[0], 1)).cuda().float(),
                 'depth': depth}
    return params, variables


def get_depth_gaussians(params, variables):
    """
    Get the parameters of the depth gaussians.
    """
    params_depth = {}
    for key in params.keys():
        params_depth[key] = params[key][variables['depth'] == 1]

    return params_depth


def initialize_optimizer(params, variables, configs):
    lrs = {
        'means3D': 0.00016 * variables['scene_radius'], #original: 0.00016 * variables['scene_radius'],
        #'rgb_colors': 0.00025, # 0.025,
        'seg_colors': 0.0,
        'unnorm_rotations': 0.001,
        'logit_opacities': 0.05,
        'log_scales': 0.005,
        'shs_dc': 0.0025,
        'shs_rest': 0.000125
    }
    param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items()]
    return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)


def get_loss(
        i,
        params, 
        curr_data,
        variables, 
        is_initial_timestep, 
        configs,
        explicit_depth=False, 
        depth_pt_cld=None, 
        grad_depth=None,
        density_mean=None,
        transmittance_mean=None,
        grad_transmittance=None,
        finite_element_transmittance=None,
        depth_smoothness=False,
        alpha_zero_one=False):
    losses = {}

    rendervar = params2rendervar(params)
    rendervar['means2D'].retain_grad()

    im, radius, depth = Renderer(raster_settings=curr_data['cam'])(**rendervar)

    curr_id = curr_data['id']
    losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))
    variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification
    
    # Add depth component to loss if depth point cloud is available        
    if explicit_depth:
        indices_gaussians_depth = torch.nonzero(variables['depth'])[:, 0]
        means_depth = params['means3D'][indices_gaussians_depth]
        losses['depth'] = chamfer_distance(means_depth.unsqueeze(0), depth_pt_cld.unsqueeze(0))[0]

    if grad_depth is not None:
        losses['grad_depth'] = grad_depth

    if density_mean is not None:
        losses['density_mean'] = -density_mean

    if transmittance_mean is not None:
        losses['transmittance'] = transmittance_mean

    if grad_transmittance is not None:
        losses['grad_transmittance'] = -grad_transmittance
    
    if finite_element_transmittance is not None:
        losses['finite_element_transmittance'] = -finite_element_transmittance

    if depth_smoothness == True and i>5000:
        losses['depth_smoothness'] = utils_gaussian.edge_aware_smoothness_per_pixel(
            curr_data['im'].unsqueeze(0), depth.unsqueeze(0).clip(0,10), i
        )

    if alpha_zero_one == True:
        losses['alpha_zero_one'] = utils_gaussian.alpha_zero_one(alpha)

    loss_weights = {'im': 1.0, 'seg': 3.0, 'rigid': 4.0, 'rot': 4.0, 'iso': 2.0,
                    'floor': 2.0, 'bg': 20.0, 'soft_col_cons': 0.01, 
                    'depth': configs['lambda_explicit_depth'],
                    'grad_depth': configs['lambda_grad_depth'],
                    'density_mean': configs['lambda_density_mean'],
                    'transmittance': configs['lambda_transmittance'],
                    'grad_transmittance': configs['lambda_grad_transmittance'],
                    'finite_element_transmittance': configs['lambda_finite_element_transmittance'],
                    'depth_smoothness': configs['lambda_depth_smoothness'],
                    'alpha_zero_one': configs['lambda_alpha_zero_one']
    }

    loss = sum([loss_weights[k] * v for k, v in losses.items()])
    if i % 200 == 0:
        print([f'{k}: {(loss_weights[k] * v):.6f}' for k, v in losses.items()])
    seen = radius > 0
    variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
    variables['seen'] = seen
    return loss, variables


def initialize_per_timestep(params, variables, optimizer):
    pts = params['means3D']
    rot = torch.nn.functional.normalize(params['unnorm_rotations'])
    new_pts = pts + (pts - variables["prev_pts"])   # smart way to initialise: new initial position
                                                    # is the previous position + the difference between
                                                    # the previous position and the position before that
    new_rot = torch.nn.functional.normalize(rot + (rot - variables["prev_rot"]))

    is_fg = params['seg_colors'][:, 0] > 0.5
    prev_inv_rot_fg = rot[is_fg]
    prev_inv_rot_fg[:, 1:] = -1 * prev_inv_rot_fg[:, 1:]
    fg_pts = pts[is_fg]
    prev_offset = fg_pts[variables["neighbor_indices"]] - fg_pts[:, None]
    variables['prev_inv_rot_fg'] = prev_inv_rot_fg.detach()
    variables['prev_offset'] = prev_offset.detach()
    variables["prev_col"] = params['rgb_colors'].detach()
    variables["prev_pts"] = pts.detach()
    variables["prev_rot"] = rot.detach()

    new_params = {'means3D': new_pts, 'unnorm_rotations': new_rot}
    params = update_params_and_optimizer(new_params, params, optimizer)

    return params, variables


def report_progress(params, data, i, progress_bar, variables, every_i=100):
    if i % every_i == 0:
        im, _, _ = Renderer(raster_settings=data['cam'])(**params2rendervar(params))
        curr_id = data['id']
        psnr = calc_psnr(torch.clamp((im), 0, 1), data['im']).mean()
        
        progress_bar.set_postfix({"train img 0 PSNR": f"{psnr:.{7}f}",
                                  "G": variables['depth'].shape[0],
                                  "DG": torch.sum(variables['depth']==1).item()})
        progress_bar.update(every_i)


def save_eval_output_data(input_seq, exp_name, output_seq):
    """Save output data for evaluation."""
    rgb_pred_npy, depth_pred_npy, pc_pred_npy = extract_output_data.extract_output_data(
        input_seq, exp_name, output_seq
    )                         
    extract_output_data.save_output_pointcloud(pc_pred_npy, exp_name, output_seq)
    extract_output_data.save_output_depth_images(depth_pred_npy, exp_name, output_seq)
    extract_output_data.save_output_rgb_images(rgb_pred_npy, exp_name, output_seq)


def update_lr_means3D(optimizer, i, scene_radius, iterations):
    ''' Learning rate scheduling per step '''
    for param_group in optimizer.param_groups:
        if param_group["name"] == "means3D":
            lr = means_scheduler_args(i, lr_init=0.00016*scene_radius, lr_final=0.0000016*scene_radius, max_steps=iterations)
            param_group['lr'] = lr
            break


def main(configs):#seq, exp, output_seq, args):

    output_dir = os.path.join(
        os.path.dirname(output.__file__),
        configs['exp_name'],
        configs['output_seq']
    )
    if os.path.exists(output_dir):
        print(f'Output folder {configs["exp_name"]}/{configs["output_seq"]} already exists. Exiting.')
        return
    
    dataset_dir = utils_data.get_dataset_dir(configs['dataset'])
    configs['input_seq'] = os.path.join(dataset_dir, configs['input_seq'])
    
    md = json.load(open(f"{os.path.dirname(data.__file__)}/{configs['input_seq']}/train_meta.json", 'r'))  # metadata
    num_timesteps = len(md['fn'])
    params, variables = initialize_params(configs['input_seq'], md)
    
    # Params and variables for depth gaussians
    depth_pt_cld = None
    transmittance_mean = None
    grad_transmittance = None
    finite_element_transmittance = None
    if configs['explicit_depth'] or configs['density'] or configs['grad_depth'] or configs['transmittance'] or configs['grad_transmittance'] or configs['finite_element_transmittance']:
        params_depth_init, variables_depth_init, depth_pt_cld = initialise_depth_gaussians(
            configs['input_seq'], md, configs['num_touches'], configs['random_selection'], configs['dataset']
        )

        # Combine params and variables for normal and depth gaussians
        params, variables = combine_params_and_variables(params, params_depth_init, variables, variables_depth_init)

    grad_depth = None
    density_mean = None
    # if configs['grad_depth'] or configs['grad_transmittance'] or configs['finite_element_transmittance']:
    #     normals = utils_mesh.estimate_normals(depth_pt_cld)
    #     # Debug normals by visualising them using plotly. Normals are visualised as vectors.
    #     utils_mesh._debug_normals(depth_pt_cld, normals)    

    optimizer = initialize_optimizer(params, variables, configs)
    output_params = []

    for t in range(num_timesteps):    
        dataset = get_dataset(t, md, configs)
        todo_dataset = []
        is_initial_timestep = (t == 0)
        if not is_initial_timestep:
            params, variables = initialize_per_timestep(params, variables, optimizer)
        num_iter_per_timestep = configs['iterations'] if is_initial_timestep else 2000
        progress_bar = tqdm(range(num_iter_per_timestep), desc=f"timestep {t}")

        # Actual training loop. Parameters are optimised here.
        for i in range(num_iter_per_timestep):
            curr_data = get_batch(todo_dataset, dataset)

            update_lr_means3D(optimizer, i, variables['scene_radius'], configs['iterations'])
               
            # Optimise depth gaussians depending on the method
            if configs['density']:
                density_mean = utils_gaussian.calculate_density(
                    params,
                    depth_pt_cld,
                    variables
                )

            if configs['grad_depth']:
                # Calculate density for depth gaussians
                grad_depth = utils_gaussian.calculate_gradient(
                    params, 
                    depth_pt_cld,
                    normals,
                    variables,
                    utils_gaussian.calculate_density_NN
                )

            transmittance_mean = None
            if configs['transmittance']:
                transmittance_mean = utils_gaussian.calculate_transmittance(
                    params,
                    depth_pt_cld,
                    variables
                )

            if configs['grad_transmittance']:
                grad_transmittance = utils_gaussian.calculate_gradient(
                    params, 
                    depth_pt_cld,
                    normals,
                    variables,
                    utils_gaussian.calculate_transmittance
                )

            if configs['finite_element_transmittance']:
                finite_element_transmittance = utils_gaussian.finite_element_transmittance(
                    params, 
                    depth_pt_cld,
                    normals,
                    variables,
                    utils_gaussian.calculate_transmittance,
                    transmittance_mean
                )

            if configs['explicit_depth']:
                # TODO improve efficiency: Opacity is not optimised for depth gaussians
                indices_gaussians_depth = torch.nonzero(variables['depth'])[:, 0]
                params['logit_opacities'].grad[indices_gaussians_depth] = 0.0

            loss, variables = get_loss(
                i,
                params,
                curr_data,
                variables,
                is_initial_timestep,
                configs,
                depth_pt_cld=depth_pt_cld,
                grad_depth=grad_depth,
                density_mean=density_mean,
                transmittance_mean=transmittance_mean,
                grad_transmittance=grad_transmittance,
                finite_element_transmittance=finite_element_transmittance,
                depth_smoothness=configs['depth_smoothness'],
                alpha_zero_one=configs['alpha_zero_one']
            )
            loss.backward()

            with torch.no_grad():
                report_progress(params, dataset[0], i, progress_bar, variables)
                if is_initial_timestep:
                    params, variables = densify(params, variables, optimizer, i, configs['explicit_depth'], configs['iterations_densify'])
                optimizer.step()
                optimizer.zero_grad()

        progress_bar.close()
        output_params.append(params2cpu(params, is_initial_timestep))

    if os.path.exists(f"{os.path.dirname(output.__file__)}/{configs['exp_name']}/{configs['output_seq']}"):
        print(f"Experiment {configs['exp_name']} for sequence {configs['input_seq']} already exists. Exiting.")
        return
    save_config(configs)
    save_params(output_params, configs['output_seq'], configs['exp_name'])
    save_variables(variables, configs['output_seq'], configs['exp_name'])
    save_eval_helper(configs['input_seq'], configs['output_seq'], configs['exp_name'])

    # Extract data for evaluation
    if configs['save_eval_data']:
        save_eval_output_data(configs['input_seq'], configs['exp_name'], configs['output_seq'])

    if configs['eval']:
        evaluator = Evaluator(
            configs['dataset'],
            configs['exp_name'],
            configs['output_seq'],
            configs['save_eval_data']
        )
        evaluator.run_evaluation()


if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(config.__file__), "train.yaml")
    configs = read_config(config_path)

    s = time.time()
    for sequence in [configs['input_seq']]:
        main(configs) # sequence, exp_name, output_seq, args)
        torch.cuda.empty_cache()
    print(f"Total time: {time.time() - s:.2f}s")
