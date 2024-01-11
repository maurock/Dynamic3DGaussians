"""Script to export .obj file from .blend file."""
import bpy
import os

blend_file_path = bpy.data.filepath
directory = os.path.dirname(blend_file_path)
obj_name = os.path.basename(blend_file_path).replace('.blend', '.obj')
target_file = os.path.join(directory, 'obj', obj_name)
bpy.ops.export_scene.obj(
    filepath=target_file,
    axis_forward='Y',
    axis_up='Z')