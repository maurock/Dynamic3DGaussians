import bpy
import math
import mathutils
import numpy as np
from mathutils import Matrix
from PIL import Image

########## SETUP VARIABLES #############################################
render_engine = 'CYCLES'
resolution_x = 600
resolution_y = 400
cycles_samples = 100
num_cameras = 50
radius = 10
near = 2
far = 20

output_img_path = '/home/mauro/Documents/BlenderProjects/Reflective_3DGS/toaster_refl_gt2/'

# Specify the name of your Scene Collection
collection_name = "Toaster"
###########################################################################

######## DEFINE FUNCTIONS ##################################################
# Function to generate equidistant points on a hemisphere
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

# Modify the setup_depth_compositor function
def setup_depth_compositor():    
    
    # Set up the nodes in the compositor
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree

    # Clear existing nodes
    for node in tree.nodes:
        tree.nodes.remove(node)

    # Create Render Layers Node
    render_layers = tree.nodes.new('CompositorNodeRLayers')
    
    # Create Output Node
    composite = tree.nodes.new(type="CompositorNodeComposite")
    
    # Create Viewer Node
    viewer_node = tree.nodes.new(type="CompositorNodeViewer")
    viewer_node.use_alpha = False

    # Link nodes
    tree.links.new(render_layers.outputs['Depth'], composite.inputs[0])
    tree.links.new(render_layers.outputs['Depth'], viewer_node.inputs[0])
    
    return tree, render_layers, composite, viewer_node

# Normalize depth values in Python
def normalize_depth(depth_map, near, far):
    # Normalize depth values based on near and far clipping planes
    normalized_depth = (depth_map - near) / (far - near)
    normalized_depth = np.clip(normalized_depth, 0, 1)  # Clip values to 0-1 range
    
    return normalized_depth


def depth_to_point_cloud(depth_map, fx, fy, cx, cy):
    k = np.array([ [fx, 0, cx], [0, fy, cy], [0, 0, 1] ]) 
    invk = np.linalg.inv(k)
    
    bpy.context.view_layer.update()
    
    # get c2w  
    c2w = bpy.context.object.matrix_world
    rotation_matrix = Matrix.Rotation(math.pi, 4, 'X')
    c2w = c2w @ rotation_matrix
    c2w = np.asarray(c2w)
    
    def_pix = np.stack(np.meshgrid(np.arange(resolution_x) + 0.5, np.arange(resolution_y) + 0.5, 1), -1).reshape(-1, 3)
    pix_ones = np.ones((resolution_y * resolution_x, 1))
    
    radial_depth = depth_map.reshape(-1)

    # def_rays is the unnormalised rays in the camera frame!
    # x_2D = K @ x_3D, so x_3D = K^-1 @ x_2D. In this case, x_2D is pixels and x_3D is rays.
    def_rays = (invk @ def_pix.T).T
    # I DO NOT NEED TO NORMALISE RAYS FOR SOME REASON
    # rays * depth = 3D points in camera coords
    pts_cam = def_rays * radial_depth[:, None]
    z_depth = pts_cam[:, 2]
    pts4 = np.concatenate((pts_cam, pix_ones), 1)
    # pts is points in 3D world coords
    pts = (c2w @ pts4.T).T[:, :3]  
    
    # Shuffle and filter  
    random_indices = np.random.choice(pts.shape[0], size=pts.shape[0], replace=False)
    pts = pts[random_indices]

    # Retain only 10% of the array
    pts = pts[:int(0.05 * len(pts))]
    
    return pts
###############################################################################

# Set rendering variables
bpy.context.scene.render.engine = render_engine

# Set the render resolution
bpy.context.scene.render.resolution_x = resolution_x
bpy.context.scene.render.resolution_y = resolution_y

# Set the number of rendering samples (optional)
bpy.context.scene.cycles.samples = cycles_samples

## Create a new sphere
bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=(0, 0, 0))
sphere = bpy.context.object
sphere.hide_render = True

# List of coordinates on the sphere
points = fibonacci_hemisphere(num_cameras, radius)

cameras_data = []
pts_all = []

tree, render_layers, composite, viewer_node = setup_depth_compositor()

######## COLLECT CAMERA POSES ##################################################
# Create N cameras
for i in range(num_cameras):

    # Create a new camera
    bpy.ops.object.camera_add(location=(0, 0, 0))
        
    camera = bpy.context.object
            
    camera_data = camera.data
    
    # Set the camera's clipping planes
    camera_data.clip_start = 0
    camera_data.clip_end = 100
    
    # Set the camera's location
    camera.location = (points[i][0], points[i][1], points[i][2])
    
     # Calculate the direction vector from the camera's location to the origin
    direction = mathutils.Vector((0, 0, 0)) - camera.location

    # Calculate the rotation quaternion to point the camera in the desired direction
    rotation_quaternion = direction.to_track_quat('-Z', 'Y')

    # Apply the rotation to the camera
    camera.rotation_euler = rotation_quaternion.to_euler()
    
    # IMPORTANT! This applies the changes, 
    # otherwise data will not be correct
    bpy.context.view_layer.update()
    
    bpy.context.scene.camera = camera  # Set the newly created camera as the active camera
    
    # Render the scene
    bpy.ops.render.render()

    # Update scene to register new nodes
    bpy.context.view_layer.update()

   # Inside the camera loop, after rendering
    pixels = bpy.data.images['Viewer Node'].pixels 
    depth_map = np.array(pixels[:]).reshape(resolution_y, resolution_x, 4)
    depth_map = np.flipud(depth_map[:,:,0])  # Extract depth values and flip

    # Depth to point cloud
    camera_data = camera.data
    fx = camera_data.lens * resolution_x / camera_data.sensor_width
    fy = camera_data.lens * resolution_y / camera_data.sensor_height
    cx = resolution_x / 2
    cy = resolution_y / 2
    
    pts_all.append(depth_to_point_cloud(depth_map, fx, fy, cx, cy))
    
    # Normalize the depth values using your function
    normalized_depth_map = normalize_depth(depth_map, near, far)

    # Convert to 8-bit image and save
    depth_image = Image.fromarray((normalized_depth_map * 255).astype(np.uint8))
    depth_image.save(f"{output_img_path}/ims/{i}/depth.png")
    

pts_all = np.concatenate(pts_all, axis=0)
np.save(f"{output_img_path}/pointcloud_gt.npy", pts_all)