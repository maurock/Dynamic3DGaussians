
import torch
import numpy as np
import open3d as o3d
import time
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from helpers import setup_camera, quat_mult
from external import build_rotation
from colormap import colormap
from copy import deepcopy
import os
import data
import json
import time

"""
Visualiser. This method was modified from the original code for better interactivity.
- Press 'C' to change camera
- Press 'M' to change mode (color or depth)
"""

torch.cuda.empty_cache()

# RENDER_MODE = 'color'  # 'color', 'depth' or 'centers'
RENDER_MODE = 'colors'  # 'color', 'depth' or 'centers'
# RENDER_MODE = 'centers'  # 'color', 'depth' or 'centers'

ADDITIONAL_LINES = None  # None, 'trajectories' or 'rotations'
# ADDITIONAL_LINES = 'trajectories'  # None, 'trajectories' or 'rotations'
# ADDITIONAL_LINES = 'rotations'  # None, 'trajectories' or 'rotations'

# REMOVE_BACKGROUND = False  # False or True
REMOVE_BACKGROUND = True  # False or True

FORCE_LOOP = False  # False or True
# FORCE_LOOP = True  # False or True

w, h = 1280, 720
near, far = 0.0001, 1000.0
view_scale = 1 # 3.9
fps = 20
traj_frac = 25  # 4% of points
traj_length = 15
def_pix = torch.tensor(
    np.stack(np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5, 1), -1).reshape(-1, 3)).cuda().float()
pix_ones = torch.ones(h * w, 1).cuda().float()


def init_camera(y_angle=0., center_dist=2.4, cam_height=1.3, f_ratio=0.82):
    ry = y_angle * np.pi / 180
    w2c = np.array([[np.cos(ry), 0., -np.sin(ry), 0.],
                    [0.,         1., 0.,          cam_height],
                    [np.sin(ry), 0., np.cos(ry),  center_dist],
                    [0.,         0., 0.,          1.]])
    k = np.array([[f_ratio * w, 0, w / 2], [0, f_ratio * w, h / 2], [0, 0, 1]])
    return w2c, k


def load_scene_data(seq, exp, seg_as_col=False):
    params = dict(np.load(f"./output/{exp}/{seq}/params.npz"))
    params = {k: torch.tensor(v).cuda().float() for k, v in params.items()}
    is_fg = params['seg_colors'][:, 0] > 0.5
    scene_data = []
    for t in range(len(params['means3D'])):
        rendervar = {
            'means3D': params['means3D'][t],
            'colors_precomp': params['rgb_colors'][t] if not seg_as_col else params['seg_colors'],
            'rotations': torch.nn.functional.normalize(params['unnorm_rotations'][t]),
            'opacities': torch.sigmoid(params['logit_opacities']),
            'scales': torch.exp(params['log_scales']),
            'means2D': torch.zeros_like(params['means3D'][0], device="cuda")
        }
        if REMOVE_BACKGROUND:
            rendervar = {k: v[is_fg] for k, v in rendervar.items()}
        scene_data.append(rendervar)
    if REMOVE_BACKGROUND:
        is_fg = is_fg[is_fg]
    return scene_data, is_fg


def make_lineset(all_pts, cols, num_lines):
    linesets = []
    for pts in all_pts:
        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, np.float64))
        lineset.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(cols, np.float64))
        pt_indices = np.arange(len(lineset.points))
        line_indices = np.stack((pt_indices, pt_indices - num_lines), -1)[num_lines:]
        lineset.lines = o3d.utility.Vector2iVector(np.ascontiguousarray(line_indices, np.int32))
        linesets.append(lineset)
    return linesets


def calculate_trajectories(scene_data, is_fg):
    in_pts = [data['means3D'][is_fg][::traj_frac].contiguous().float().cpu().numpy() for data in scene_data]
    num_lines = len(in_pts[0])
    cols = np.repeat(colormap[np.arange(len(in_pts[0])) % len(colormap)][None], traj_length, 0).reshape(-1, 3)
    out_pts = []
    for t in range(len(in_pts))[traj_length:]:
        out_pts.append(np.array(in_pts[t - traj_length:t + 1]).reshape(-1, 3))
    return make_lineset(out_pts, cols, num_lines)


def calculate_rot_vec(scene_data, is_fg):
    in_pts = [data['means3D'][is_fg][::traj_frac].contiguous().float().cpu().numpy() for data in scene_data]
    in_rotation = [data['rotations'][is_fg][::traj_frac] for data in scene_data]
    num_lines = len(in_pts[0])
    cols = colormap[np.arange(num_lines) % len(colormap)]
    inv_init_q = deepcopy(in_rotation[0])
    inv_init_q[:, 1:] = -1 * inv_init_q[:, 1:]
    inv_init_q = inv_init_q / (inv_init_q ** 2).sum(-1)[:, None]
    init_vec = np.array([-0.1, 0, 0])
    out_pts = []
    for t in range(len(in_pts)):
        cam_rel_qs = quat_mult(in_rotation[t], inv_init_q)
        rot = build_rotation(cam_rel_qs).cpu().numpy()
        vec = (rot @ init_vec[None, :, None]).squeeze()
        out_pts.append(np.concatenate((in_pts[t] + vec, in_pts[t]), 0))
    return make_lineset(out_pts, cols, num_lines)


def render(w2c, k, timestep_data):
    with torch.no_grad():
        cam = setup_camera(w, h, k, w2c, near, far)
        im, _, depth, = Renderer(raster_settings=cam)(**timestep_data)
        return im, depth


def rgbd2pcd(im, depth, w2c, k, show_depth=False, project_to_cam_w_scale=None):
    d_near = 0.1
    d_far = 7
    invk = torch.inverse(torch.tensor(k).cuda().float())
    c2w = torch.inverse(torch.tensor(w2c).cuda().float())
    radial_depth = depth[0].reshape(-1)
    def_rays = (invk @ def_pix.T).T
    def_radial_rays = def_rays / torch.linalg.norm(def_rays, ord=2, dim=-1)[:, None]
    pts_cam = def_radial_rays * radial_depth[:, None]
    z_depth = pts_cam[:, 2]
    if project_to_cam_w_scale is not None:
        pts_cam = project_to_cam_w_scale * pts_cam / z_depth[:, None]
    pts4 = torch.concat((pts_cam, pix_ones), 1)
    pts = (c2w @ pts4.T).T[:, :3]
    if show_depth:
        cols = ((z_depth - d_near) / (d_far - d_near))[:, None].repeat(1, 3)
    else:
        cols = torch.permute(im, (1, 2, 0)).reshape(-1, 3)
    pts = o3d.utility.Vector3dVector(pts.contiguous().double().cpu().numpy())
    cols = o3d.utility.Vector3dVector(cols.contiguous().double().cpu().numpy())
    return pts, cols


def change_camera(vis, camera_positions, camera_index):
    if camera_index[0] < len(camera_positions) - 1:
        camera_index[0] += 1
    else:
        camera_index[0] = 0

    w2c, k = camera_positions[camera_index[0]]
    cam_params = vis.get_view_control().convert_to_pinhole_camera_parameters()
    cam_params.extrinsic = w2c
    vis.get_view_control().convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)
    return False

def toggle_mode(vis, mode):
    if mode[0] == 'colors':
        mode[0] = 'depth'
    elif mode[0] == 'depth':
        mode[0] = 'colors'
    vis.update_renderer()  # Update the renderer to reflect the new mode
    return False


def get_camera_positions(seq):
    path = os.path.join(os.path.dirname(data.__file__), seq, 'train_meta_interpolated.json')
    with open(path, 'r') as file:
        cameras = json.load(file)

    camera_positions = []
    for i in range(np.array(cameras["w2c"]).shape[1]):
        k = np.array(cameras["k"])[0, 0, :, :]
        w2c = np.array(cameras["w2c"])[0, i, :, :]
        camera_positions.append([w2c, k])
    
    return camera_positions

def visualize(seq, exp):
    scene_data, is_fg = load_scene_data(seq, exp)

    # With callback
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=int(w * view_scale), height=int(h * view_scale), visible=True)
    camera_index = [0]
    mode = [RENDER_MODE]
    camera_positions = get_camera_positions(seq)
    vis.register_key_callback(ord('C'), lambda vis: change_camera(vis, camera_positions, camera_index)) # Bind 'C' key to toggle camera
    vis.register_key_callback(ord('M'), lambda vis: toggle_mode(vis, mode))  # Bind 'M' key to toggle mode

    w2c, k = init_camera()

    im, depth = render(w2c, k, scene_data[0])
    init_pts, init_cols = rgbd2pcd(im, depth, w2c, k, show_depth=(mode[0] == 'depth'))
    pcd = o3d.geometry.PointCloud()
    pcd.points = init_pts
    pcd.colors = init_cols
    vis.add_geometry(pcd)

    linesets = None
    lines = None
    if ADDITIONAL_LINES is not None:
        if ADDITIONAL_LINES == 'trajectories':
            linesets = calculate_trajectories(scene_data, is_fg)
        elif ADDITIONAL_LINES == 'rotations':
            linesets = calculate_rot_vec(scene_data, is_fg)
        lines = o3d.geometry.LineSet()
        lines.points = linesets[0].points
        lines.colors = linesets[0].colors
        lines.lines = linesets[0].lines
        vis.add_geometry(lines)

    view_k = k * view_scale
    view_k[2, 2] = 1
    view_control = vis.get_view_control()
    cparams = o3d.camera.PinholeCameraParameters()
    cparams.extrinsic = w2c
    cparams.intrinsic.intrinsic_matrix = view_k
    cparams.intrinsic.height = int(h * view_scale)
    cparams.intrinsic.width = int(w * view_scale)
    view_control.convert_from_pinhole_camera_parameters(cparams, allow_arbitrary=True)

    render_options = vis.get_render_option()
    render_options.point_size = view_scale
    render_options.light_on = False

    path = os.path.join(os.path.dirname(data.__file__), seq, 'train_meta_interpolated.json')
    with open(path, 'r') as file:
        cameras = json.load(file)

    frame_counter = 0

    start_time = time.time()
    num_timesteps = len(scene_data)
    while True:
        passed_time = time.time() - start_time
        passed_frames = passed_time * fps
        if ADDITIONAL_LINES == 'trajectories':
            t = int(passed_frames % (num_timesteps - traj_length)) + traj_length  # Skip t that don't have full traj.
        else:
            t = int(passed_frames % num_timesteps)

        change_camera(vis, camera_positions, camera_index)

        print(camera_index[0])
        if (camera_index[0] == 199) or (camera_index[0] == 299) or (camera_index[0] == 399) or (camera_index[0] == 499):
            print('CIAO')
            print(camera_index[0])
            time.sleep(3)

        if (camera_index[0] == 599):
            time.sleep(10)

        if FORCE_LOOP:
            num_loops = 1.4
            y_angle = 360*t*num_loops / num_timesteps
            w2c, k = init_camera(y_angle)
            cam_params = view_control.convert_to_pinhole_camera_parameters()
            cam_params.extrinsic = w2c
            view_control.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)
        else:  # Interactive control
            cam_params = view_control.convert_to_pinhole_camera_parameters()
            view_k = cam_params.intrinsic.intrinsic_matrix
            k = view_k / view_scale

            k[2, 2] = 1
            w2c = cam_params.extrinsic


        if mode[0] == 'centers':
            pts = o3d.utility.Vector3dVector(scene_data[t]['means3D'].contiguous().double().cpu().numpy())
            cols = o3d.utility.Vector3dVector(scene_data[t]['colors_precomp'].contiguous().double().cpu().numpy())
        else:
            im, depth = render(w2c, k, scene_data[t])
            pts, cols = rgbd2pcd(im, depth, w2c, k, show_depth=(mode[0] == 'depth'))
        pcd.points = pts
        pcd.colors = cols
        vis.update_geometry(pcd)

        if ADDITIONAL_LINES is not None:
            if ADDITIONAL_LINES == 'trajectories':
                lt = t - traj_length
            else:
                lt = t
            lines.points = linesets[lt].points
            lines.colors = linesets[lt].colors
            lines.lines = linesets[lt].lines
            vis.update_geometry(lines)


        if not vis.poll_events():
            break
        vis.update_renderer()

        frame_path = f"output/exp1/earth/frames/frame_{frame_counter}.png"
        vis.capture_screen_image(frame_path)
        frame_counter += 1

    vis.destroy_window()
    del view_control
    del vis
    del render_options


if __name__ == "__main__":
    exp_name = "exp1"
    for sequence in ["earth"]: #, "boxes", "football", "juggle", "softball", "tennis"]:
        visualize(sequence, exp_name)
