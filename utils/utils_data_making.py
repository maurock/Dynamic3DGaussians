from PIL import Image as PIL_Image
import os

def generate_seg_images(output_root_dir, rgb_folder, num_cam, img_name):
    """Generate black images PNG as everything is static currently. Images have the same size as the original images.
    TODO: change this to generate segmentation images when doing dynamic scenes.
    Parameters:
        rgb_folder (str): Path to the folder that contains rendering image.
        num_cam (str): Number of the camera.
        output_root_dir (str): Path to the output root folder.
    """
    original_image = PIL_Image.open(os.path.join(rgb_folder, img_name))
    width, height = original_image.size
    black_image = PIL_Image.new('L', (width, height), 0)
    seg_folder = os.path.join(output_root_dir, 'seg')
    target_seg_folder = os.path.join(seg_folder, num_cam)
    if not os.path.exists(target_seg_folder):
        os.makedirs(target_seg_folder)
    black_image.save(os.path.join(target_seg_folder,'render.png'))