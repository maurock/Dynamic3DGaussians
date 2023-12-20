
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
import matplotlib.cm as cm
import plotly.graph_objects as go
import output


"""
Visualiser. This method was modified from the original code for better interactivity.
- Press 'C' to change camera
- Press 'M' to change mode (color or depth)
"""

torch.cuda.empty_cache()

RENDER_MODE = 'color'  # 'color', 'depth' or 'centers'
# RENDER_MODE = 'color'  # 'color', 'depth' or 'centers'
# RENDER_MODE = 'centers'  # 'color', 'depth' or 'centers'

ADDITIONAL_LINES = None  # None, 'trajectories' or 'rotations'
# ADDITIONAL_LINES = 'trajectories'  # None, 'trajectories' or 'rotations'
# ADDITIONAL_LINES = 'rotations'  # None, 'trajectories' or 'rotations'

# REMOVE_BACKGROUND = False  # False or True
REMOVE_BACKGROUND = True  # False or True

FORCE_LOOP = False  # False or True
# FORCE_LOOP = True  # False or True

w, h = 600, 400 #640*2, 360*2
near, far = 0.00001, 10000.0
view_scale = 1 # 3.9
fps = 20
traj_frac = 25  # 4% of points
traj_length = 15
def_pix = torch.tensor(
    np.stack(np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5, 1), -1).reshape(-1, 3)).cuda().float()
pix_ones = torch.ones(h * w, 1).cuda().float()
colormappa = cm.magma  # You can change this to cm.jet, cm.viridis, etc.


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
    d_near = 2.0
    d_far = 15.0
    invk = torch.inverse(torch.tensor(k).cuda().float())
    c2w = torch.inverse(torch.tensor(w2c).cuda().float())
    print(depth.shape)
    radial_depth = depth[0].reshape(-1)
    # def_rays is the unnormalised rays in the camera frame
    # x_2D = K @ x_3D, so x_3D = K^-1 @ x_2D. In this case, x_2D is pixels and x_3D is rays.
    def_rays = (invk @ def_pix.T).T
    # normalise
    def_radial_rays = def_rays / torch.linalg.norm(def_rays, ord=2, dim=-1)[:, None]
    # rays * depth = 3D points in camera coords
    pts_cam = def_radial_rays * radial_depth[:, None]
    z_depth = pts_cam[:, 2]
    if project_to_cam_w_scale is not None:
        pts_cam = project_to_cam_w_scale * pts_cam / z_depth[:, None]
    pts4 = torch.concat((pts_cam, pix_ones), 1)
    # pts is points in 3D world coords
    pts = (c2w @ pts4.T).T[:, :3]
    if show_depth:
        cols = ((z_depth - d_near) / (d_far - d_near))[:, None].repeat(1, 3)
    else:
        cols = torch.permute(im, (1, 2, 0)).reshape(-1, 3)

    # Filter 
    custom_mn = np.array([-1.2, -1.2, -1])
    custom_mx = np.array([1.2, 1.2, 1])
    pts = pts.contiguous().double().cpu().numpy()
    cols = cols.contiguous().double().cpu().numpy()
    mask = (pts < custom_mx).all(-1) & (pts > custom_mn).all(-1)
    pts_masked = pts[mask]
    cols_masked = cols[mask]

    pts = o3d.utility.Vector3dVector(pts_masked)
    cols = o3d.utility.Vector3dVector(cols_masked)

    # return pts, cols

    # Estimate normals
    # Create a point cloud object

    pcd = o3d.geometry.PointCloud()
    pcd.points = pts
    pcd.colors = cols

    # Estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Flip normals to point outward
    camera_location = np.array([0, 0, 0])  # Assuming camera at origin in its coordinate system
    camera_location = np.dot(np.linalg.inv(w2c[:3, :3]), camera_location - w2c[:3, 3])  # Transform to world coordinates

    for i, normal in enumerate(pcd.normals):
        point = np.asarray(pcd.points)[i]
        vector_to_camera = camera_location - point
        dot_product = np.dot(normal, vector_to_camera)
        if dot_product < 0:
            pcd.normals[i] = -normal

    return pcd.points, pcd.colors, pcd.normals


def change_camera(vis, camera_positions, camera_index, pts_all, pts):
    if camera_index[0] < len(camera_positions) - 1:
        camera_index[0] += 1
    else:
        camera_index[0] = 0

    w2c, k = camera_positions[camera_index[0]]
    cam_params = vis.get_view_control().convert_to_pinhole_camera_parameters()
    cam_params.extrinsic = w2c
    vis.get_view_control().convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)
    pts_all.append(np.asarray(pts))

    return pts_all

def get_camera_positions(seq, cameras):
    camera_positions = []
    for i in range(np.array(cameras["k"]).shape[1]):
        k = np.array(cameras["k"])[0, i, :, :]
        w2c = np.array(cameras["w2c"])[0, i, :, :]
        camera_positions.append([w2c, k])
    
    return camera_positions, k, w2c

def visualize(seq, exp):
    scene_data, is_fg = load_scene_data(seq, exp)

    path = os.path.join(os.path.dirname(data.__file__), seq, 'train_meta.json')
    with open(path, 'r') as file:
        cameras = json.load(file)

    # With callback
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=int(w * view_scale), height=int(h * view_scale), visible=True)
    camera_index = [0]
    mode = [RENDER_MODE]
    camera_positions, k, w2c = get_camera_positions(seq, cameras)

    im, depth = render(w2c, k, scene_data[0])
    init_pts, init_cols, _ = rgbd2pcd(im, depth, w2c, k, show_depth=(mode[0] == 'depth'))
    pcd = o3d.geometry.PointCloud()
    pcd.points = init_pts
    pcd.colors = init_cols
    vis.add_geometry(pcd)

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

    start_time = time.time()
    num_timesteps = len(scene_data)

    # Accumulate all points for plotting
    pts_all = []
    normals_all = []

    for i in range(np.array(cameras["k"]).shape[1]):
        passed_time = time.time() - start_time
        passed_frames = passed_time * fps

        t = int(passed_frames % num_timesteps)

        cam_params = view_control.convert_to_pinhole_camera_parameters()
        view_k = cam_params.intrinsic.intrinsic_matrix
        k = view_k / view_scale
        k[2, 2] = 1
        w2c = cam_params.extrinsic

        im, depth = render(w2c, k, scene_data[t])
        pts, cols, normals = rgbd2pcd(im, depth, w2c, k, show_depth=(mode[0] == 'depth'))
        
        # Accumulate all points for plotting
        pts_all = change_camera(vis, camera_positions, camera_index, pts_all, pts)
        normals_all.append(normals)
            
        pcd.points = pts
        pcd.colors = cols
        vis.update_geometry(pcd)

        if not vis.poll_events():
            break
        vis.update_renderer()

    vis.destroy_window()
    del view_control
    del vis
    del render_options
    
    #### DEBUG
    # Plot pts with plotly
    pts_all = np.concatenate(pts_all, axis=0)
    normals_all = np.concatenate(normals_all, axis=0)

    custom_mn = np.array([-1.2, -1.2, -1])
    custom_mx = np.array([1.2, 1.2, 1])
    mask = (pts_all < custom_mx).all(-1) & (pts_all > custom_mn).all(-1)
    pts_all = pts_all[mask]
    normals_all = normals_all[mask]

    # Reduce it to 10% of the points
    # Shuffle the array
    random_indices = np.random.choice(pts_all.shape[0], size=pts_all.shape[0], replace=False)
    pts_all = pts_all[random_indices]
    normals_all = normals_all[random_indices]

    # Retain only 10% of the array
    pts_all = pts_all[:int(0.05 * len(pts_all))]
    normals_all = normals_all[:int(0.05 * len(normals_all))]

    output_path = f"{os.path.dirname(output.__file__)}/{exp}/{seq}/{seq}.npy"
    np.save(output_path, pts_all)

    # fig = go.Figure(data=[go.Scatter3d(
    #     x=pts_all[:,0],
    #     y=pts_all[:,1],
    #     z=pts_all[:,2],
    #     mode='markers',
    #     marker=dict(
    #         size=2
    #     )
    # )])
    # fig.add_trace(go.Cone(
    #     x=pts_all[:, 0],
    #     y=pts_all[:, 1],
    #     z=pts_all[:, 2],
    #     u=normals_all[:, 0],
    #     v=normals_all[:, 1],
    #     w=normals_all[:, 2],
    #     sizemode="absolute",
    #     sizeref=1,
    #     anchor="tail"
    # ))
    # fig.show()
    # ###

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pts_all)
    # pcd.normals = o3d.utility.Vector3dVector(normals_all)

    # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    #     mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    #         pcd, depth=9)
    #     o3d.io.write_triangle_mesh('toaster_refl_transparency_FE.obj', mesh, compressed=True, print_progress=True)


if __name__ == "__main__":
    exp_name = "exp1"
    for sequence in ["toaster_refl_transparency_FE"]: #["toaster_refl", "toaster_refl_transparency", "toaster_refl_transparency_FE", "toaster_refl_FE"]: 
        visualize(sequence, exp_name)
