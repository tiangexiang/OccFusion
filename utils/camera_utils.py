#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
import cv2
import torch
WARNED = False



def calculate_elevation_azimuth_radius(R, T):
    """
    Calculate the elevation, azimuth, and radius from camera extrinsic parameters.

    Parameters:
    - R: Rotation matrix (3x3 numpy array)
    - T: Translation vector (3x1 numpy array)

    Returns:
    - elevation: The elevation angle in radians
    - azimuth: The azimuth angle in radians
    - radius: The distance from the camera to the origin
    """
    # Camera position in world coordinates, C = -R^-1 * T
    C = -np.linalg.inv(R).dot(T)

    # Radius = magnitude of C
    radius = np.linalg.norm(C)

    # Azimuth = angle of projection of C onto the X-Y plane with X-axis
    azimuth = np.arctan2(C[1], C[0])

    # Elevation = angle between C and its projection onto the X-Y plane
    elevation = np.arcsin(C[2] / radius)

    return np.rad2deg(elevation), np.rad2deg(azimuth), np.rad2deg(radius)

# # Example usage:
# R = np.array([[0.707, -0.707, 0], [0.707, 0.707, 0], [0, 0, 1]]) # Example rotation matrix
# T = np.array([1, 2, 3]) # Example translation vector

# elevation, azimuth, radius = calculate_elevation_azimuth_radius(R, T)
# elevation_deg = np.degrees(elevation)  # Convert elevation to degrees
# azimuth_deg = np.degrees(azimuth)  # Convert azimuth to degrees


def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 3200:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    if cam_info.bound_mask is not None:
        resized_bound_mask = PILtoTorch(cam_info.bound_mask, resolution)
    else:
        resized_bound_mask = None

    if cam_info.bkgd_mask is not None:
        resized_bkgd_mask = PILtoTorch(cam_info.bkgd_mask, resolution)
    else:
        resized_bkgd_mask = None

    if cam_info.gen_msk is not None:
        resized_gen_msk = PILtoTorch(cam_info.gen_msk, resolution)
    else:
        resized_gen_msk = None

    if cam_info.gen_image is not None:
        resized_gen_image = PILtoTorch(cam_info.gen_image, resolution)
    else:
        resized_gen_image = None

    if cam_info.human_mask is not None:
        resized_human_mask = PILtoTorch(cam_info.human_mask, resolution)
    else:
        resized_human_mask = None

    # if cam_info.has_occ is not None:
    #     has_occ = cam_info.has_occ
    # else:
    #     has_occ = None


    return Camera(colmap_id=cam_info.uid, pose_id=cam_info.pose_id, R=cam_info.R, T=cam_info.T, K=cam_info.K, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, bkgd_mask=resized_bkgd_mask,
                  gen_msk=resized_gen_msk,
                  gen_image=resized_gen_image,
                  human_mask=resized_human_mask, 
                  bound_mask=resized_bound_mask, smpl_param=cam_info.smpl_param, 
                  world_vertex=cam_info.world_vertex, world_bound=cam_info.world_bound, 
                  big_pose_smpl_param=cam_info.big_pose_smpl_param, 
                  big_pose_world_vertex=cam_info.big_pose_world_vertex, 
                  big_pose_world_bound=cam_info.big_pose_world_bound, 
                  openpose=cam_info.openpose, 
                  data_device=args.data_device)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry

def _update_extrinsics(
        extrinsics, 
        angle, 
        trans=None, 
        rotate_axis='y'):
    r""" Uptate camera extrinsics when rotating it around a standard axis.

    Args:
        - extrinsics: Array (3, 3)
        - angle: Float
        - trans: Array (3, )
        - rotate_axis: String

    Returns:
        - Array (3, 3)
    """
    if torch.is_tensor(extrinsics):
        extrinsics = extrinsics.detach().cpu().numpy()

    E = extrinsics
    inv_E = np.linalg.inv(E)

    camrot = inv_E[:3, :3]
    campos = inv_E[:3, 3]
    if trans is not None:
        campos -= trans

    rot_y_axis = camrot.T[1, 1]
    if rot_y_axis < 0.:
        angle = -angle
    
    rotate_coord = {
        'x': 0, 'y': 1, 'z':2
    }
    grot_vec = np.array([0., 0., 0.])
    grot_vec[rotate_coord[rotate_axis]] = angle
    grot_mtx = cv2.Rodrigues(grot_vec)[0].astype('float32')

    rot_campos = grot_mtx.dot(campos) 
    rot_camrot = grot_mtx.dot(camrot)
    if trans is not None:
        rot_campos += trans
    
    new_E = np.identity(4)
    new_E[:3, :3] = rot_camrot.T
    new_E[:3, 3] = -rot_camrot.T.dot(rot_campos)

    return new_E

def rotate_camera_by_frame_idx(
        extrinsics, 
        frame_idx, 
        trans=None,
        rotate_axis='y',
        period=196,
        inv_angle=False):
    r""" Get camera extrinsics based on frame index and rotation period.

    Args:
        - extrinsics: Array (3, 3)
        - frame_idx: Integer
        - trans: Array (3, )
        - rotate_axis: String
        - period: Integer
        - inv_angle: Boolean (clockwise/counterclockwise)

    Returns:
        - Array (3, 3)
    """

    angle = 2 * np.pi * (frame_idx / period)
    if inv_angle:
        angle = -angle
    return _update_extrinsics(
                extrinsics, angle, trans, rotate_axis)