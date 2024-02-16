import trimesh
import data
import os
import helpers

def main():
    ply_dir = os.path.join(
        os.path.dirname(data.__file__), 'meshes-pretrained', 'syn'
    )
    output_dir = os.path.join(
        os.path.dirname(data.__file__), 'meshes-pretrained', 'obj'
    )
    helpers.create_dirs(output_dir)
    objs = os.listdir(ply_dir)

    for obj in objs:
        ply_path = os.path.join(ply_dir, obj)
        mesh = trimesh.load_mesh(ply_path)
        obj_path = os.path.join(output_dir, obj.replace('.ply', '.obj'))
        
        mesh.export(obj_path, file_type='obj')


if __name__=='__main__':
    main()