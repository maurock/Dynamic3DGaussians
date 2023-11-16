import bpy
import math
import mathutils
from random import random, choice
import os
import json
import bmesh
import numpy as np
from mathutils import Matrix

########## SETUP VARIABLES #############################################
# These are the only variables you need to set.
render_engine = 'CYCLES'
resolution_x = 600
resolution_y = 400
cycles_samples = 100
num_cameras = 50
radius = 10
sky_texture_path = 'path/to/sky_texture'

output_img_path = 'path/to/ims'   # this is the folder where images are saved, please call it 'ims'
output_poses_path = 'path/to/cameras_gt.json' # this is the folder where cameras info are stored, please call it 'cameras_gt.json'
output_point_path = 'path/to/init_pt_cld.npz' # this is the folder where cameras info are stored, please call it 'init_pt_cld.npz'

# Specify the name of your Scene Collection
collection_name = "NAME_OF_YOUR_COLLECTION"
n_points_to_sample = 3000  # per mesh
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

def make_serialisable(obj):
    return [list(row) for row in obj]

# Function to generate a random point on a triangle
def random_point_in_triangle(v1, v2, v3):
    r1, r2 = random(), random()
    sqrt_r1 = math.sqrt(r1)

    u = 1 - sqrt_r1
    v = sqrt_r1 * (1 - r2)
    w = r2 * sqrt_r1

    return (u * v1.co) + (v * v2.co) + (w * v3.co)

# Function to generate a random point on a polygon
def random_point_in_polygon(verts):
    # Triangulate the polygon, the first vertex is shared by all triangles
    triangles = [(verts[0], verts[i], verts[i + 1]) for i in range(1, len(verts) - 1)]

    # Choose a random triangle
    chosen_triangle = choice(triangles)

    # Generate a random point in the chosen triangle
    return random_point_in_triangle(*chosen_triangle)


# Function to get the material color
def get_material_color(obj, face):
    mat_color = [1, 1, 1]  # Default color (white) in case no material is found
    if obj.material_slots and face.material_index < len(obj.material_slots):
        mat = obj.material_slots[face.material_index].material
        if mat and mat.node_tree:
            for node in mat.node_tree.nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    base_color = node.inputs['Base Color'].default_value
                    mat_color = [base_color[0], base_color[1], base_color[2]]  # RGB values
                    break
    return mat_color
###############################################################################

# Set up sky
# Get the environment node tree of the current scene
node_tree = bpy.context.scene.world.node_tree
tree_nodes = node_tree.nodes

# Clear all nodes
tree_nodes.clear()

# Add Background node
node_background = tree_nodes.new(type='ShaderNodeBackground')

# Add Environment Texture node
node_environment = tree_nodes.new('ShaderNodeTexEnvironment')
# Load and assign the image to the node property
node_environment.image = bpy.data.images.load(sky_texture_path) 
node_environment.location = -300,0

# Add Output node
node_output = tree_nodes.new(type='ShaderNodeOutputWorld')   
node_output.location = 200,0

# Link all nodes
links = node_tree.links
link = links.new(node_environment.outputs["Color"], node_background.inputs["Color"])
link = links.new(node_background.outputs["Background"], node_output.inputs["Surface"])

bpy.context.scene.render.film_transparent = False

# Set rendering variables
bpy.context.scene.render.engine = render_engine

# Set the render resolution
bpy.context.scene.render.resolution_x = resolution_x
bpy.context.scene.render.resolution_y = resolution_y

# Set file format
bpy.context.scene.render.image_settings.file_format = 'JPEG'

# Set the number of rendering samples (optional)
bpy.context.scene.cycles.samples = cycles_samples

## Create a new sphere
bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=(0, 0, 0))
sphere = bpy.context.object
sphere.hide_render = True

# List of coordinates on the sphere
points = fibonacci_hemisphere(num_cameras, radius)

cameras_data = []

######## COLLECT CAMERA POSES ##################################################
# Create N cameras
for i in range(num_cameras):

    # Create a new camera
    bpy.ops.object.camera_add(location=(0, 0, 0))
        
    camera = bpy.context.object
    
    camera_data = camera.data
    
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
    
    # get c2w and w2c    
    c2w = bpy.context.object.matrix_world
    # Rotate matrix by 180 degrees around its Local x-axis, to make sure that the z-axis points inward.
    # IMPORTANT: This is required by the repository I am using.
    rotation_matrix = Matrix.Rotation(math.pi, 4, 'X')
    c2w = c2w @ rotation_matrix
    w2c = c2w.inverted()
    
    # Make everything serialisable
    w2c = make_serialisable(w2c)
    c2w = make_serialisable(c2w)
    
    bpy.context.scene.camera = camera  # Set the newly created camera as the active camera

    # Store pose cameras into dict
    single_camera = dict()    
    
    single_camera['w2c'] = w2c
    single_camera['c2w'] = c2w
    single_camera['id'] = i
    single_camera['img_name'] = f'render_{i}'
    single_camera['width'] = resolution_x
    single_camera['height'] = resolution_y
    single_camera['position'] = list(camera.location)
    single_camera['fx'] = camera_data.lens * resolution_x / camera_data.sensor_width
    single_camera['fy'] = camera_data.lens * resolution_y / camera_data.sensor_height
    
    cameras_data.append(single_camera)
    
    # Render the image
    # Set the output path for the rendered image
    bpy.context.scene.render.filepath = f"{output_img_path}/{i}/render.jpg"
    bpy.ops.render.render(write_still=True)

with open(output_poses_path, 'w') as json_file:
    json.dump(cameras_data, json_file)
#############################################################################################    
    
###### SAMPLE POINTS FROM MESH ###############################################################
# Ensure you're in object mode
bpy.ops.object.mode_set(mode='OBJECT')

# Initialize lists to store points and colors
means3D = []
rgb_colors = []
seg = []

# Get the collection
collection = bpy.data.collections.get(collection_name)
if not collection:
    print(f"No collection found by the name '{collection_name}'")
else:
    # Iterate through each object in the collection
    for obj in collection.objects:
        if obj.type == 'MESH':
            # Create a BMesh from the object
            bm = bmesh.new()
            bm.from_mesh(obj.data)
            bm.transform(obj.matrix_world)
            
            # Update the BMesh's internal index table
            bm.faces.ensure_lookup_table()
            bm.verts.ensure_lookup_table()
            bm.edges.ensure_lookup_table()
                
            # Sample points
            for _ in range(n_points_to_sample):  # replace 100 with the number of samples you want

                # Find a random face
                face = bm.faces[int(random() * len(bm.faces))]
                # Calculate a random point on the face
                # point = face.calc_center_median()
                # Check if the face is a triangle or a polygon
                if len(face.verts) == 3:
                    # If triangle, directly use the vertices
                    point = random_point_in_triangle(*face.verts)
                else:
                    # If polygon, first triangulate
                    point = random_point_in_polygon(face.verts)
                means3D.append(point.to_tuple())  # Store point
                
                print(obj.name, point)
                
                # Get the color of the material of the face
                mat_color = get_material_color(obj, face)
                rgb_colors.append(mat_color)  

            # Free the BMesh
            bm.free()

# Concatenate means3D, rgb_colors and seg
means3D = np.array(means3D)
rgb_colors = np.array(rgb_colors)
seg = np.ones(shape=(rgb_colors.shape[0], 1))
sampled_data = dict()
sampled_data['data'] = np.concatenate((means3D, rgb_colors, seg), axis=1)

# Save the dictionary as a .npz file
np.savez_compressed(output_point_path, **sampled_data)