from PIL import Image as PIL_Image
import os

def generate_seg_images(args, target_folder, num_cam, img_name):
    """Generate black images PNG as everything is static currently. Images have the same size as the original images.
    TODO: change this to generate segmentation images when doing dynamic scenes.
    Parameters:
        target_folder (str): Path to the folder where the original images are.
        num_cam (str): Number of the camera.
    """
    original_image = PIL_Image.open(os.path.join(target_folder, img_name))
    width, height = original_image.size
    black_image = PIL_Image.new('L', (width, height), 0)
    seg_folder = os.path.join(args.output_path, args.dataset_name, 'seg')
    target_seg_folder = os.path.join(seg_folder, num_cam)
    if not os.path.exists(target_seg_folder):
        os.makedirs(target_seg_folder)
    black_image.save(os.path.join(target_seg_folder,'render.png'))