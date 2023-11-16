"""
Utilities to extract camera parameters from COLMAP reconstruction files and convert it into the required format.
Some code is adapted from https://github.com/graphdeco-inria/gaussian-splatting
"""

import struct
import numpy as np
import collections
import os
import sys
from PIL import Image as PIL_Image
from utils import utils_math
from typing import NamedTuple


CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])

# From https://github.com/graphdeco-inria/gaussian-splatting
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
}

# From https://github.com/graphdeco-inria/gaussian-splatting
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])

# From https://github.com/graphdeco-inria/gaussian-splatting
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])

# From https://github.com/graphdeco-inria/gaussian-splatting
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])

# From https://github.com/graphdeco-inria/gaussian-splatting
def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

# From https://github.com/graphdeco-inria/gaussian-splatting
class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

# From https://github.com/graphdeco-inria/gaussian-splatting
def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


# From https://github.com/graphdeco-inria/gaussian-splatting
def read_extrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images

# From https://github.com/graphdeco-inria/gaussian-splatting
def read_intrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    print(path_to_model_file)
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8*num_params,
                                     format_char_sequence="d"*num_params)
            cameras[camera_id] = Camera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras

# From https://github.com/graphdeco-inria/gaussian-splatting
def read_points3D_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """


    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]

        xyzs = np.empty((num_points, 3))
        rgbs = np.empty((num_points, 3))
        errors = np.empty((num_points, 1))

        for p_id in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid, num_bytes=8*track_length,
                format_char_sequence="ii"*track_length)
            xyzs[p_id] = xyz
            rgbs[p_id] = rgb
            errors[p_id] = error
    return xyzs, rgbs, errors


def get_intrinsics_matrix(extr, intr):
    """Get intrinsics matrix for single timestep. 
    Params:
        extr: dict of extrinsics
        intr: dict of intrinsics
    Returns:
        ks: (N, 3, 3) where N is the number of unique cameras
    """
    # Index of valid images. Some images are discarded by COLMAP
    img_idxs = sorted(list(extr.keys()))
    
    # List of intrinsics matrices
    ks = []
    
    for img_idx in img_idxs:

        # Extrincs of current image
        im = extr[img_idx]

        # Camera ID of current image
        cam_id = im.camera_id

        # Intrinsics of current image
        cam = intr[cam_id]
        assert cam_id == cam.id

        if cam.model == 'SIMPLE_PINHOLE':
            w = cam.width
            h = cam.height
            fx = cam.params[0]
            fy = cam.params[0]
            cx = cam.params[1]
            cy = cam.params[2]

        elif cam.model == 'PINHOLE':
            # Not tested
            w = cam.width
            h = cam.height
            fx = cam.params[0]
            fy = cam.params[1]
            cx = cam.params[2]
            cy = cam.params[3]

        k = [ [fx, 0, cx], [0, fy, cy], [0, 0, 1] ]
        ks.append(k)
   
    return ks

# From https://github.com/graphdeco-inria/gaussian-splatting
class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int


# From https://github.com/graphdeco-inria/gaussian-splatting
def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = utils_math.focal2fov(focal_length_x, height)
            FovX = utils_math.focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = utils_math.focal2fov(focal_length_y, height)
            FovX = utils_math.focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
        
        cam_valid_idx = 0
        while not os.path.exists(os.path.join(images_folder, str(cam_valid_idx), os.path.basename(extr.name))):
            cam_valid_idx += 1
            
        image_path = os.path.join(images_folder, str(cam_valid_idx), os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = PIL_Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def get_extrinsics_matrix(imgs_path, extr, intr):
    # Get info about all the cameras
    cams = readColmapCameras(extr, intr, imgs_path)

    extr_matrices = []
    # cams are already ordered by image id, e.g. images/0/render.png, images/1/render.png
    for cam in cams:
        # Convert to 4x4 matrix
        Rt = np.eye(4)
        Rt[:3, :3] = cam.R.transpose()
        Rt[:3, 3] = cam.T

        # w2c = np.linalg.inv(Rt)
        extr_matrices.append(Rt.tolist())

    return extr_matrices


def get_cam_images(extr):
    """Get image path as <cam_id>/<img_file> and cam ids as list"""
    img_idxs = sorted(list(extr.keys()))

    images = []
    cam_ids = []
    for idx in img_idxs:
        images.append(extr[idx].name)
        cam_ids.append(extr[idx].name.split('/')[0])
    
    return images, cam_ids

        


