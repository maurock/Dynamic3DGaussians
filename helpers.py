import torch
import os
import open3d as o3d
import numpy as np
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from PIL import Image
import yaml
import cv2
import output
import matplotlib.pyplot as plt
import output

def setup_camera(w, h, k, w2c, near=0.01, far=100):
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    w2c = torch.tensor(w2c).cuda().float()
    cam_center = torch.inverse(w2c)[:3, 3]
    w2c = w2c.unsqueeze(0).transpose(1, 2)
    opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                                [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                                [0.0, 0.0, 1.0, 0.0]]).cuda().float().unsqueeze(0).transpose(1, 2)
    full_proj = w2c.bmm(opengl_proj)
    cam = Camera(
        image_height=h,
        image_width=w,
        tanfovx=w / (2 * fx),
        tanfovy=h / (2 * fy),
        bg=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=w2c,
        projmatrix=full_proj,
        sh_degree=6,
        campos=cam_center,
        prefiltered=False,
        debug=False
    )
    return cam


def params2rendervar(params):
    """This converts the parameters to variables necessary for rendering.
    Specifically:
    - rotations are normalised
    - opacities are processed by a sigmoid to be between [0, 1]
    - scales are converted from log_scales    
    """
    rendervar = {
        'means3D': params['means3D'],
        'colors_precomp': params['rgb_colors'],
        'rotations': torch.nn.functional.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(params['log_scales']),   # maybe because some gaussians are very big and some very small? 
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0
    }
    return rendervar


def l1_loss_v1(x, y):
    return torch.abs((x - y)).mean()


def l1_loss_v2(x, y):
    return (torch.abs(x - y).sum(-1)).mean()


def weighted_l2_loss_v1(x, y, w):
    return torch.sqrt(((x - y) ** 2) * w + 1e-20).mean()


def weighted_l2_loss_v2(x, y, w):
    return torch.sqrt(((x - y) ** 2).sum(-1) * w + 1e-20).mean()


def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1.T
    w2, x2, y2, z2 = q2.T
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z]).T


def o3d_knn(pts, num_knn):
    indices = []
    sq_dists = []
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, np.float64))
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    for p in pcd.points:
        [_, i, d] = pcd_tree.search_knn_vector_3d(p, num_knn + 1)
        indices.append(i[1:])
        sq_dists.append(d[1:])
    return np.array(sq_dists), np.array(indices)


def params2cpu(params, is_initial_timestep):
    if is_initial_timestep:
        res = {k: v.detach().cpu().contiguous().numpy() for k, v in params.items()}
    else:
        res = {k: v.detach().cpu().contiguous().numpy() for k, v in params.items() if
               k in ['means3D', 'rgb_colors', 'unnorm_rotations']}
    return res


def save_params_static(to_save, output_params, seq, exp):
    """Use if I only have 1 timestamp"""
    print('Saving parameters for evaluation... Only works for static scenes.')
    for k in output_params[0].keys():
        # Increase dimensionality by 1 to account for timesteps
        if k in ['means3D', 'rgb_colors', 'unnorm_rotations']:
            to_save[k] = [output_params[0][k]]
        else:
            to_save[k] = output_params[0][k]

    os.makedirs(f"{os.path.dirname(output.__file__)}/{exp}/{seq}", exist_ok=True)
    np.savez(f"{os.path.dirname(output.__file__)}/{exp}/{seq}/params", **to_save)
    print('Parameters saved.')


def save_params(output_params, seq, exp):
    to_save = {}
    if len(output_params) == 1:
        save_params_static(to_save, output_params, seq, exp)
    else:
        for k in output_params[0].keys():
            if k in output_params[1].keys():
                to_save[k] = np.stack([params[k] for params in output_params])
            else:
                to_save[k] = output_params[0][k]
        os.makedirs(f"{os.path.dirname(output.__file__)}/{exp}/{seq}", exist_ok=True)
        np.savez(f"{os.path.dirname(output.__file__)}/{exp}/{seq}/params", **to_save)


def save_variables(output_variables, seq, exp):
    print('Saving variables for evaluation... Only works for static scenes.')
    to_save = {}
    for k, v in output_variables.items():
        to_save[k] = v.detach().cpu() if isinstance(v, torch.Tensor) else v
    os.makedirs(f"{os.path.dirname(output.__file__)}/{exp}/{seq}", exist_ok=True)
    np.savez(f"{os.path.dirname(output.__file__)}/{exp}/{seq}/variables", **to_save)
    print('Variables saved.')


def save_eval_helper(input_seq, output_seq, exp):
    """Save the object name to a text file for evaluation"""
    if not os.path.exists(f"{os.path.dirname(output.__file__)}/{exp}/{output_seq}/eval"):
        os.makedirs(f"{os.path.dirname(output.__file__)}/{exp}/{output_seq}/eval")
    with open(f"{os.path.dirname(output.__file__)}/{exp}/{output_seq}/eval/eval_helper.txt", "w") as f:
        f.write(input_seq)


def load_rgb_image(path):
    """Load image from path and convert to torch tensor of shape (3, H, W)"""
    img_np = np.array(Image.open(path))
    return torch.tensor(img_np).float().cuda().permute(2, 0, 1) / 255


def load_disparity_image(path):
    """Load disparity image from path (H, W, 4) and convert to torch tensor of shape (H, W).
    The disparity image is stored as a 16-bit per-channel TIFF file. As a standard procedure,
    because we are decoding the raw bytes of the image, we need to convert the bytes to a numpy array
    and normalise it by dividing by 2^16."""

    with open(path, 'rb') as f:
        bytes_ = np.asarray(bytearray(f.read()), dtype=np.uint8)

    disp = np.array(cv2.imdecode(bytes_, cv2.IMREAD_UNCHANGED), dtype=np.float32)
    disp = disp / 2**16
    disp_filtered = disp[:, : , 0] # only use the first channel

    return torch.tensor(disp_filtered).float().cuda()


def convert_disparity_to_depth(disparity_img):
    """Convert disparity to depth: 
    disparity = 1 / (1 + depth), so depth = 1 / disp - 1"""
    depth = 1 / (disparity_img + 1e-9) - 1
    return depth


def normalise_depth(depth, min_depth, max_depth):
    """Clip depth between two vlaues and normalise it.
    Parameters:
        depth (np.array | torch.tensor): Depth image, shape (H, W)
    """
    # Clip depth between min and max depth (np.array or torch.Tensor)
    if isinstance(depth, np.ndarray):
        depth = np.clip(depth, min_depth, max_depth)
    elif isinstance(depth, torch.Tensor):
        depth = torch.clamp(depth, min_depth, max_depth)
    depth = (depth - min_depth) / (max_depth - min_depth)
    return depth 
    

def read_config(config_path):
    """Read the YAML config file and return dictionary of parameters"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config):
    """Save the config dictionary to a YAML file"""
    path = os.path.join(
        os.path.dirname(output.__file__),
        config['exp_name'],
        config['output_seq'],
        'config.yaml'
    )
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'w') as f:
        yaml.dump(config, f)


def create_dirs(dirs):
    """Create required directories if they do not already exist.
    It accepts single directory or a list of directories."""
    if not isinstance(dirs, list):
        dirs = [dirs]
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir) 


def render(w, h, k, w2c, near, far, timestep_data,train=False):
    with torch.no_grad():
        cam = setup_camera(w, h, k, w2c, near, far)
        im, radius, depth, alpha = Renderer(raster_settings=cam,train=train)(**timestep_data)
        return im, depth, alpha
        # im, radius, depth = Renderer(raster_settings=cam)(**timestep_data)
        # return im, depth
    
def debug_smoothness_loss(
        img, 
        pred,
        pred_gradients_x, pred_gradients_y,
        image_gradients_x, image_gradients_y,
        weights_x, weights_y,
        smoothness_x, smoothness_y
    ):
    """
    img: shape (1, 3, h, w)
    pred: shape (1, 1, h, w)
    pred_gradients_x/y: shape (1, 1, h, w)
    image_gradients_x/y: shape (1, 3, h, w)
    weights_x/y: shape (1, 1, h, w)
    smoothness_x/y: shape (1, 1, h, w)
    
    Debug edge_aware_smoothness_per_pixel function"""

    fig = plt.figure(figsize=(20, 40))
    fig.add_subplot(1, 6, 1)
    plt.imshow(img[0].permute(1, 2, 0).detach().cpu())
    plt.title('RGB Image X')
    fig.add_subplot(1, 6, 2)
    plt.imshow(pred[0, 0].detach().cpu())
    plt.title('Predicted Depth')
    fig.add_subplot(1, 6, 3)
    plt.imshow(pred_gradients_x[0, 0].detach().cpu())
    plt.title('Predicted Depth Gradient X')
    fig.add_subplot(1, 6, 4)
    plt.imshow(image_gradients_x[0].permute(1, 2, 0).detach().cpu())
    plt.title('Image Gradient X')
    fig.add_subplot(1, 6, 5)
    plt.imshow(weights_x[0, 0].detach().cpu())
    plt.title('Weights X')
    fig.add_subplot(1, 6, 6)
    plt.imshow(smoothness_x[0, 0].detach().cpu())
    plt.title('Smoothness X')
    plt.show()

    fig = plt.figure(figsize=(20, 40))
    fig.add_subplot(1, 6, 1)
    plt.imshow(img[0].permute(1, 2, 0).detach().cpu())
    plt.title('RGB Image Y')
    fig.add_subplot(1, 6, 2)
    plt.imshow(pred[0, 0].detach().cpu())
    plt.title('Predicted Depth')
    fig.add_subplot(1, 6, 3)
    plt.imshow(pred_gradients_y[0, 0].detach().cpu())
    plt.title('Predicted Depth Gradient')
    fig.add_subplot(1, 6, 4)
    plt.imshow(image_gradients_y[0].permute(1, 2, 0).detach().cpu())
    plt.title('Image Gradient')
    fig.add_subplot(1, 6, 5)
    plt.imshow(weights_y[0, 0].detach().cpu())
    plt.title('Weights')
    fig.add_subplot(1, 6, 6)
    plt.imshow(smoothness_y[0, 0].detach().cpu())
    plt.title('Smoothness')
    plt.show()






    
    
