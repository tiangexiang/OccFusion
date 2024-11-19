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

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from utils.pose_utils import draw_poses
import numpy as np
import torch
import json
import imageio
import cv2
import random
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from utils.camera_utils import _update_extrinsics
from scene.gaussian_model import BasicPointCloud
import pickle

from smpl.smpl_numpy import SMPL
from smplx import SMPL as SMPLPT
from smplx.body_models import SMPLX

#from data.dna_rendering.dna_rendering_sample_code.SMCReader import SMCReader
import trimesh
from scene.render_utils import (smpl_to_pose, PerspectiveCamera, Renderer, render_trimesh, \
                                estimate_translation_cv2, transform_smpl)
class CameraInfo(NamedTuple):
    uid: int
    pose_id: int
    R: np.array
    T: np.array
    K: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    bkgd_mask: np.array
    human_mask: np.array
    bound_mask: np.array
    width: int
    height: int
    smpl_param: dict
    world_vertex: np.array
    world_bound: np.array
    big_pose_smpl_param: dict
    big_pose_world_vertex: np.array
    big_pose_world_bound: np.array
    openpose: np.array
    gen_msk: np.array
    gen_image: np.array

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

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
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames[:20]):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, bkgd_mask=None, bound_mask=None, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


##################################   MonoCap   ##################################
def get_camera_extrinsics_monocap(view_index, val=False, camera_view_num=36):
    def norm_np_arr(arr):
        return arr / np.linalg.norm(arr)

    def lookat(eye, at, up):
        zaxis = norm_np_arr(at - eye)
        xaxis = norm_np_arr(np.cross(zaxis, up))
        yaxis = np.cross(xaxis, zaxis)
        _viewMatrix = np.array([
            [xaxis[0], xaxis[1], xaxis[2], -np.dot(xaxis, eye)],
            [yaxis[0], yaxis[1], yaxis[2], -np.dot(yaxis, eye)],
            [-zaxis[0], -zaxis[1], -zaxis[2], np.dot(zaxis, eye)],
            [0       , 0       , 0       , 1     ]
        ])
        return _viewMatrix
    
    def fix_eye(phi, theta):
        camera_distance = 3
        return np.array([
            camera_distance * np.sin(theta) * np.cos(phi),
            camera_distance * np.sin(theta) * np.sin(phi),
            camera_distance * np.cos(theta)
        ])

    if val:
        at = np.array([0, 0.8, 0]).astype(np.float32)
        eye = fix_eye(np.pi + np.pi/12 + 1e-6, -np.pi/2 + 2 * np.pi * view_index / camera_view_num + 1e-6).astype(np.float32) + at

        extrinsics = lookat(eye, at, np.array([0, 1, 0])).astype(np.float32)
    return extrinsics

def readCamerasMonoCapdata(path, output_view, white_background, image_scaling=1.0, split='train', novel_view_vis=False):
    cam_infos = []

    if 'olek_images0812' in path or 'vlad_images1011' in path:
        pose_start = 1
    else:
        pose_start = 0
    if split == 'train':
        pose_interval = 5
        pose_num = 100
    elif split == 'test':
        pose_interval = 30
        pose_num = 17 

    annot_path = os.path.join(path, 'annots.npy')
    annots = np.load(annot_path, allow_pickle=True).item()
    cam = annots['cams']

    # load SMPL model
    smpl_model = SMPL(sex='neutral', model_dir='assets/SMPL_NEUTRAL_renderpeople.pkl')

    # SMPL in canonical space
    big_pose_smpl_param = {}
    big_pose_smpl_param['R'] = np.eye(3).astype(np.float32)
    big_pose_smpl_param['Th'] = np.zeros((1,3)).astype(np.float32)
    big_pose_smpl_param['shapes'] = np.zeros((1,10)).astype(np.float32)
    big_pose_smpl_param['poses'] = np.zeros((1,72)).astype(np.float32)
    big_pose_smpl_param['poses'][0, 5] = 45/180*np.array(np.pi)
    big_pose_smpl_param['poses'][0, 8] = -45/180*np.array(np.pi)
    big_pose_smpl_param['poses'][0, 23] = -30/180*np.array(np.pi)
    big_pose_smpl_param['poses'][0, 26] = 30/180*np.array(np.pi)

    big_pose_xyz, _ = smpl_model(big_pose_smpl_param['poses'], big_pose_smpl_param['shapes'].reshape(-1))
    big_pose_xyz = (np.matmul(big_pose_xyz, big_pose_smpl_param['R'].transpose()) + big_pose_smpl_param['Th']).astype(np.float32)

    # obtain the original bounds for point sampling
    big_pose_min_xyz = np.min(big_pose_xyz, axis=0)
    big_pose_max_xyz = np.max(big_pose_xyz, axis=0)
    big_pose_min_xyz -= 0.05
    big_pose_max_xyz += 0.05
    big_pose_world_bound = np.stack([big_pose_min_xyz, big_pose_max_xyz], axis=0)

    idx = 0
    for pose_index in range(pose_start, pose_start+pose_num*pose_interval, pose_interval):
        for view_index in output_view:

            if novel_view_vis:
                view_index_look_at = view_index
                view_index = 0

            # Load image, mask, K, D, R, T
            if 'olek_images0812' in path:
                image_path = os.path.join(path, 'images', str(view_index).zfill(2), str(pose_index).zfill(6)+'.jpg')
                msk_path = os.path.join(path, 'mask', str(view_index).zfill(2), str(pose_index).zfill(6)+'.png')
            elif 'vlad_images1011' in path:
                image_path = os.path.join(path, 'images', str(view_index).zfill(3), str(pose_index).zfill(6)+'.jpg')
                msk_path = os.path.join(path, 'mask', str(view_index).zfill(3), str(pose_index).zfill(6)+'.jpg')
            else:
                image_path = os.path.join(path, 'images', str(view_index).zfill(2), str(pose_index).zfill(4)+'.jpg')
                msk_path = os.path.join(path, 'mask', str(view_index).zfill(2), str(pose_index).zfill(4)+'.png')
            
            image_name = view_index
            image = np.array(imageio.imread(image_path).astype(np.float32) / 255.)

            msk = imageio.imread(msk_path).astype(np.float32) / 255
            if msk.shape[-1] == 3:
                msk = msk[:,:,0]

            if not novel_view_vis:
                cam_id = view_index
                K = cam['K'][cam_id]
                D = cam['D'][cam_id]
                R = cam["R"][cam_id]
                T = cam["T"][cam_id][...,None].reshape(-1, 1) / 1000

                # undistort image and mask 
                image = cv2.undistort(image, K, D)
                msk = cv2.undistort(msk, K, D)
            else:
                pose = np.matmul(np.array([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]]), get_camera_extrinsics_monocap(view_index_look_at, val=True))
                R = pose[:3,:3]
                T = pose[:3, 3].reshape(-1, 1)
                cam_id = view_index
                K = cam['K'][cam_id]
            
            # mask image
            if 'olek_images0812' in path or 'vlad_images1011' in path:
                image = image * msk[...,None].repeat(3, axis=2)
            else:
                image[msk == 0] = 1 if white_background else 0

            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            w2c = np.eye(4)
            w2c[:3,:3] = R
            w2c[:3,3:4] = T

            # get the world-to-camera transform and set R, T
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            # Reduce the image resolution by ratio, then remove the back ground
            ratio = image_scaling
            if ratio != 1.0:
                H, W = int(image.shape[0] * ratio), int(image.shape[1] * ratio)
                image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)
                msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
                K[:2] = K[:2] * ratio

            image = Image.fromarray(np.array(image*255.0, dtype=np.byte), "RGB")

            focalX = K[0,0]
            focalY = K[1,1]

            FovX = focal2fov(focalX, image.size[0])
            FovY = focal2fov(focalY, image.size[1])

            # load smpl data 
            params_path = os.path.join(path, 'params',
                                    '{}.npy'.format(pose_index))
            params = np.load(params_path, allow_pickle=True).item()
            Rh = params['Rh'].astype(np.float32)
            Th = params['Th'].astype(np.float32)

            smpl_param = {}
            smpl_param['shapes'] = np.array(params['shapes']).astype(np.float32)
            smpl_param['poses'] = np.array(params["poses"]).astype(np.float32).reshape(1,72)
            smpl_param['R'] = cv2.Rodrigues(Rh)[0].astype(np.float32) #np.eye(3).astype(np.float32)
            smpl_param['Th'] = Th #np.array(params["Th"]).astype(np.float32)
            xyz, _ = smpl_model(smpl_param['poses'], smpl_param['shapes'].reshape(-1))
            xyz = (np.matmul(xyz, smpl_param['R'].transpose()) + smpl_param['Th']).astype(np.float32)

            # obtain the original bounds for point sampling
            min_xyz = np.min(xyz, axis=0)
            max_xyz = np.max(xyz, axis=0)
            min_xyz -= 0.1
            max_xyz += 0.1
            world_bound = np.stack([min_xyz, max_xyz], axis=0)

            # get bounding mask and bcakground mask
            bound_mask = get_bound_2d_mask(world_bound, K, w2c[:3], image.size[1], image.size[0])

            bound_mask = Image.fromarray(np.array(bound_mask*255.0, dtype=np.byte))
            bkgd_mask = Image.fromarray(np.array(msk*255.0, dtype=np.byte))
            cam_infos.append(CameraInfo(uid=idx, pose_id=pose_index, R=R, T=T, K=K, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, bkgd_mask=bkgd_mask, 
                            bound_mask=bound_mask, width=image.size[0], height=image.size[1], 
                            smpl_param=smpl_param, world_vertex=xyz, world_bound=world_bound, 
                            big_pose_smpl_param=big_pose_smpl_param, big_pose_world_vertex=big_pose_xyz, 
                            big_pose_world_bound=big_pose_world_bound))


            idx += 1
            
    return cam_infos

def readMonoCapdataInfo(path, white_background, output_path, eval):

    if 'olek_images0812' in path:
        train_view = [44]
        test_view = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 49]
        test_view = [45]
    elif 'vlad_images1011' in path:        
        train_view = [66]
        test_view = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  
    else:
        train_view = [0]
        test_view = range(1,11)

    print("Reading Training Transforms")
    train_cam_infos = readCamerasMonoCapdata(path, train_view, white_background, split='train')
    print("Reading Test Transforms")
    test_cam_infos = readCamerasMonoCapdata(path, test_view, white_background, split='test', novel_view_vis=False)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    if len(train_view) == 1:
        nerf_normalization['radius'] = 1

    # ply_path = os.path.join(path, "points3d.ply")
    ply_path = os.path.join('output', output_path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 6890 #100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = train_cam_infos[0].big_pose_world_vertex

        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

##################################   ZJUMoCapRefine   ##################################

def get_camera_extrinsics_zju_mocap_refine(view_index, val=False, camera_view_num=36):
    def norm_np_arr(arr):
        return arr / np.linalg.norm(arr)

    def lookat(eye, at, up):
        zaxis = norm_np_arr(at - eye)
        xaxis = norm_np_arr(np.cross(zaxis, up))
        yaxis = np.cross(xaxis, zaxis)
        _viewMatrix = np.array([
            [xaxis[0], xaxis[1], xaxis[2], -np.dot(xaxis, eye)],
            [yaxis[0], yaxis[1], yaxis[2], -np.dot(yaxis, eye)],
            [-zaxis[0], -zaxis[1], -zaxis[2], np.dot(zaxis, eye)],
            [0       , 0       , 0       , 1     ]
        ])
        return _viewMatrix
    
    def fix_eye(phi, theta):
        camera_distance = 3
        return np.array([
            camera_distance * np.sin(theta) * np.cos(phi),
            camera_distance * np.sin(theta) * np.sin(phi),
            camera_distance * np.cos(theta)
        ])

    if val:
        eye = fix_eye(np.pi + 2 * np.pi * view_index / camera_view_num + 1e-6, np.pi/2 + np.pi/12 + 1e-6).astype(np.float32) + np.array([0, 0, -0.8]).astype(np.float32)
        at = np.array([0, 0, -0.8]).astype(np.float32)

        extrinsics = lookat(eye, at, np.array([0, 0, -1])).astype(np.float32)
    return extrinsics

def readCamerasZJUMoCapRefine(path, gen_root, output_view, white_background, image_scaling=0.5, split='train', novel_view_vis=False, view_nums = None, frame_nums = None):

    cam_infos = []
    occ_param = {
        '377': [460, 70],
        '386': [415, 80],
        '387': [448, 50],
        '392': [417, 78],
        '393': [434, 68],
        '394': [418, 64]
        }

    pose_start = 0
    if split == 'train':
        pose_interval = 5
        pose_num = 100
    elif split == 'test':
        pose_start = 0
        pose_interval = 30
        pose_num = 17

        ####### rendering logic
        # pose_start = 107
        # pose_interval = 5
        # pose_num = 1
        # output_view = [13]
        
    PAPER_RENDER = False

    ##### for visual 
    
    
    if view_nums is None:
        view_nums = [i for i in range(len(output_view))]
    if frame_nums is None:
        frame_nums = [i for i in range(pose_num)]

    subject = path.split('/')[-1].split('_')[-1]
    ann_file = os.path.join(path, 'annots.npy')
    annots = np.load(ann_file, allow_pickle=True).item()
    cams = annots['cams']

    smpl2op = [24,12,17,19,21,16,18,20,2,5,8,1,4,7,25,26,27,28]
    smpl_model_pt = SMPLPT(gender='neutral', model_path='assets/SMPL_NEUTRAL.pkl')

    inpainted_root = os.path.join(inpainted_root, subject, 'incontext_inpainted')


    ims = np.array([
        np.array(ims_data['ims'])[output_view]
        for ims_data in annots['ims'][pose_start:]
    ])
    mask_frame = int(ims.shape[0] * 0.8)

    print('number of frames been occluded:', mask_frame)

    ims = np.array([
        np.array(ims_data['ims'])[output_view]
        for ims_data in annots['ims'][pose_start:pose_start + pose_num * pose_interval][::pose_interval]
    ])


    cam_inds = np.array([
        np.arange(len(ims_data['ims']))[output_view]
        for ims_data in annots['ims'][pose_start:pose_start + pose_num * pose_interval][::pose_interval]
    ])

    if 'CoreView_313' in path or 'CoreView_315' in path:
        for i in range(ims.shape[0]):
            ims[i] = [x.split('/')[0] + '/' + x.split('/')[1].split('_')[4] + '.jpg' for x in ims[i]]


    smpl_model = SMPL(sex='neutral', model_dir='assets/SMPL_NEUTRAL.pkl')

    # SMPL in canonical space
    big_pose_smpl_param = {}
    big_pose_smpl_param['R'] = np.eye(3).astype(np.float32)
    big_pose_smpl_param['Th'] = np.zeros((1,3)).astype(np.float32)
    big_pose_smpl_param['shapes'] = np.zeros((1,10)).astype(np.float32)
    big_pose_smpl_param['poses'] = np.zeros((1,72)).astype(np.float32)
    big_pose_smpl_param['poses'][0, 5] = 45/180*np.array(np.pi)
    big_pose_smpl_param['poses'][0, 8] = -45/180*np.array(np.pi)
    big_pose_smpl_param['poses'][0, 23] = -30/180*np.array(np.pi)
    big_pose_smpl_param['poses'][0, 26] = 30/180*np.array(np.pi)

    big_pose_xyz, _ = smpl_model(big_pose_smpl_param['poses'], big_pose_smpl_param['shapes'].reshape(-1))
    big_pose_xyz = (np.matmul(big_pose_xyz, big_pose_smpl_param['R'].transpose()) + big_pose_smpl_param['Th']).astype(np.float32)

    # obtain the original bounds for point sampling
    big_pose_min_xyz = np.min(big_pose_xyz, axis=0)
    big_pose_max_xyz = np.max(big_pose_xyz, axis=0)
    big_pose_min_xyz -= 0.05
    big_pose_max_xyz += 0.05
    big_pose_world_bound = np.stack([big_pose_min_xyz, big_pose_max_xyz], axis=0)
    idx = 0

    
    view_nums = [0] #### only one view for loading efficiency
    for pose_index in frame_nums:
        for view_index in view_nums:

            if novel_view_vis:
                view_index_look_at = view_index
                view_index = 0

            # Load image, mask, K, D, R, T
            image_path = os.path.join(path, ims[pose_index][view_index].replace('\\', '/'))
            if split == 'test':
                print(f"Opening {image_path} for test")
            image_name = ims[pose_index][view_index].split('.')[0]
            image = np.array(imageio.imread(image_path).astype(np.float32)/255.)

            msk_path = image_path.replace('images', 'mask').replace('jpg', 'png')
            msk = imageio.imread(msk_path)
            msk_mod = msk.copy()

            if split == 'train' and pose_index < 0.8 * pose_num: #Mask out first 80% of pixels
                midpoint, width = occ_param[subject][0], occ_param[subject][1]
                msk_mod[:,midpoint-width//2:midpoint+width//2] *= 0
            else:
                msk_mod = None

            if PAPER_RENDER:
                midpoint, width = occ_param[subject][0], occ_param[subject][1]
                msk_mod = msk.copy()
                msk_mod[:,midpoint-width//2:midpoint+width//2] *= 0

            if os.path.exists(os.path.join(inpainted_root, image_name.split('/')[-1]+'.png')) and msk_mod is not None:
                gen_image = imageio.imread(os.path.join(inpainted_root, image_name.split('/')[-1]+'.png'))
                gen_image = np.array(gen_image.astype(np.float32)/255.)
                gen_image = cv2.resize(gen_image, (image.shape[0], image.shape[1]), interpolation=cv2.INTER_AREA)
            else:
                gen_image = None
            
            msk = (msk != 0).astype(np.uint8)
            if msk_mod is not None:
                msk_mod = (msk_mod != 0).astype(np.uint8)


            if not novel_view_vis:
                cam_ind = cam_inds[pose_index][view_index]
                K = np.array(cams['K'][cam_ind])
                D = np.array(cams['D'][cam_ind])
                R = np.array(cams['R'][cam_ind])
                T = np.array(cams['T'][cam_ind]) / 1000.

                image = cv2.undistort(image, K, D)
                msk = cv2.undistort(msk, K, D)
                if msk_mod is not None:
                    msk_mod = cv2.undistort(msk_mod, K, D)
                if gen_image is not None:
                    gen_image = cv2.undistort(gen_image, K, D)
            else:
                pose = np.matmul(np.array([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]]), get_camera_extrinsics_zju_mocap_refine(view_index_look_at, val=True))
                R = pose[:3,:3]
                T = pose[:3, 3].reshape(-1, 1)
                cam_ind = cam_inds[pose_index][view_index]
                K = np.array(cams['K'][cam_ind])

            if msk_mod is not None:
                image[msk_mod == 0] = 1 if white_background else 0
            else:
                image[msk == 0] = 1 if white_background else 0

            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            w2c = np.eye(4)
            w2c[:3,:3] = R
            w2c[:3,3:4] = T

            # get the world-to-camera transform and set R, T
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            # Reduce the image resolution by ratio, then remove the back ground
            ratio = image_scaling
            if ratio != 1.:
                H, W = int(image.shape[0] * ratio), int(image.shape[1] * ratio)
                image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)
                msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
                if msk_mod is not None:
                    msk_mod = cv2.resize(msk_mod, (W, H), interpolation=cv2.INTER_NEAREST)
                if gen_image is not None:
                    gen_image = cv2.resize(gen_image, (W, H), interpolation=cv2.INTER_AREA)
                K[:2] = K[:2] * ratio

            image = Image.fromarray(np.array(image*255.0, dtype=np.byte), "RGB")

            if gen_image is not None:
                gen_image = Image.fromarray(np.array(gen_image*255.0, dtype=np.byte), "RGB")

            focalX = K[0,0]
            focalY = K[1,1]
            FovX = focal2fov(focalX, image.size[0])
            FovY = focal2fov(focalY, image.size[1])

            # load smpl data 
            i = int(os.path.basename(image_path)[:-4])
            vertices_path = os.path.join(path, 'smpl_vertices', '{}.npy'.format(i))
            xyz = np.load(vertices_path).astype(np.float32)

            smpl_param_path = os.path.join(path, "smpl_params", '{}.npy'.format(i))
            smpl_param = np.load(smpl_param_path, allow_pickle=True).item()
            Rh = smpl_param['Rh']
            smpl_param['R'] = cv2.Rodrigues(Rh)[0].astype(np.float32)
            smpl_param['Th'] = smpl_param['Th'].astype(np.float32)
            smpl_param['shapes'] = smpl_param['shapes'].astype(np.float32)
            smpl_param['poses'] = smpl_param['poses'].astype(np.float32)

            min_xyz = np.min(xyz, axis=0)
            max_xyz = np.max(xyz, axis=0)
            min_xyz -= 0.05
            max_xyz += 0.05
            world_bound = np.stack([min_xyz, max_xyz], axis=0)

            # get bounding mask and bcakground mask
            bound_mask = get_bound_2d_mask(world_bound, K, w2c[:3], image.size[1], image.size[0])
            bound_mask = Image.fromarray(np.array(bound_mask*255.0, dtype=np.byte))
            final_image = Image.blend(image, bound_mask.convert('RGB'), 0.5)

            if msk_mod is not None:
                bkgd_mask = Image.fromarray(np.array(msk_mod*255.0, dtype=np.byte))
            else:
                bkgd_mask = Image.fromarray(np.array(msk*255.0, dtype=np.byte))
            human_mask = Image.fromarray(np.array(msk*255.0, dtype=np.byte))

            ##### SMPL to openpose
            output = smpl_model_pt(betas = torch.tensor(smpl_param['shapes'].reshape(-1))[None].float(), 
                                   body_pose = torch.tensor(smpl_param['poses'])[:,3:].float())
            joints3d = output.joints.detach().squeeze().cpu().numpy()
            joints3d = joints3d[smpl2op]

            joints3d = joints3d @ smpl_param['R'].T + smpl_param['Th'].astype(np.float32)

            rotate_smpl = None
            ###### rotate
            if rotate_smpl is not None:
                w2c = _update_extrinsics(w2c, rotate_smpl, smpl_param['Th'][0])

            joints3d = joints3d @ w2c[:3,:3].T + w2c[:3, [3]].T
            joints2d = joints3d @ K.T
            joints2d = joints2d[:,:2] / joints2d[:,[-1]]

            canvas = draw_poses(joints2d, image.size[1], image.size[0])
            canvas = Image.fromarray(np.uint8(canvas))


            cam_infos.append(CameraInfo(uid=idx, pose_id=pose_index, R=R, T=T, K=K, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, bkgd_mask=bkgd_mask, 
                            human_mask = human_mask,
                            bound_mask=bound_mask, width=image.size[0], height=image.size[1], 
                            smpl_param=smpl_param, world_vertex=xyz, world_bound=world_bound, 
                            big_pose_smpl_param=big_pose_smpl_param, big_pose_world_vertex=big_pose_xyz, 
                            big_pose_world_bound=big_pose_world_bound, openpose=canvas, gen_msk=human_mask, gen_image=gen_image))

            idx += 1
            
    return cam_infos

def readZJUMoCapRefineInfo(path, white_background, output_path, eval, view_nums=None, frame_nums=None):
    train_view = [0]
    test_view = [i for i in range(0, 23)]
    test_view.remove(train_view[0])

    print("Reading Training Transforms")
    train_cam_infos = readCamerasZJUMoCapRefine(path, train_view, white_background, split='train')
    print("Reading Test Transforms")
    test_cam_infos = readCamerasZJUMoCapRefine(path, test_view, white_background, split='test', novel_view_vis=False)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    if len(train_view) == 1:
        nerf_normalization['radius'] = 1

    ply_path = os.path.join('output', output_path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 6890 #100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = train_cam_infos[0].big_pose_world_vertex

        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

##################################   OcMotion   ##################################

def get_camera_extrinsics_ocmotion(view_index, val=False, camera_view_num=36):
    def norm_np_arr(arr):
        return arr / np.linalg.norm(arr)

    def lookat(eye, at, up):
        zaxis = norm_np_arr(at - eye)
        xaxis = norm_np_arr(np.cross(zaxis, up))
        yaxis = np.cross(xaxis, zaxis)
        _viewMatrix = np.array([
            [xaxis[0], xaxis[1], xaxis[2], -np.dot(xaxis, eye)],
            [yaxis[0], yaxis[1], yaxis[2], -np.dot(yaxis, eye)],
            [-zaxis[0], -zaxis[1], -zaxis[2], np.dot(zaxis, eye)],
            [0       , 0       , 0       , 1     ]
        ])
        return _viewMatrix
    
    def fix_eye(phi, theta):
        camera_distance = 3
        return np.array([
            camera_distance * np.sin(theta) * np.cos(phi),
            camera_distance * np.sin(theta) * np.sin(phi),
            camera_distance * np.cos(theta)
        ])

    if val:
        eye = fix_eye(np.pi + 2 * np.pi * view_index / camera_view_num + 1e-6, np.pi/2 + np.pi/12 + 1e-6).astype(np.float32) + np.array([0, 0, -0.8]).astype(np.float32)
        at = np.array([0, 0, -0.8]).astype(np.float32)

        extrinsics = lookat(eye, at, np.array([0, 0, -1])).astype(np.float32)
    return extrinsics

def readCamerasOcMotion(path, gen_root, output_view, white_background, image_scaling=0.5, split='train', novel_view_vis=False, rotate_smpl=None):
    #print(gen_root, white_background, rotate_smpl)
    cam_infos = []

    subject = path.split('/')[-1]
    json_file = os.path.join(path, 'metadata.json')
    with open(json_file) as f:
        annots = json.load(f)
    f.close()

    pkl_file = os.path.join(path, 'mesh_infos.pkl')
    mesh_infos = pickle.load(open(pkl_file, 'rb'))
    cam_file = os.path.join(path, 'all_cameras.pkl')
    all_cameras = pickle.load(open(cam_file, 'rb'))

    smpl_model = SMPL(sex='neutral', model_dir='assets/SMPL_NEUTRAL.pkl')
    smpl2op = [24,12,17,19,21,16,18,20,2,5,8,1,4,7,25,26,27,28]
    smpl_model_pt = SMPLPT(gender='neutral', model_path='assets/SMPL_NEUTRAL.pkl')

    # Initialize SMPL in canonical space
    big_pose_smpl_param = {}
    big_pose_smpl_param['R'] = np.eye(3).astype(np.float32)
    big_pose_smpl_param['Th'] = np.zeros((1,3)).astype(np.float32)
    big_pose_smpl_param['shapes'] = np.zeros((1,10)).astype(np.float32)
    big_pose_smpl_param['poses'] = np.zeros((1,72)).astype(np.float32)
    big_pose_smpl_param['poses'][0, 5] = 45/180*np.array(np.pi)
    big_pose_smpl_param['poses'][0, 8] = -45/180*np.array(np.pi)
    big_pose_smpl_param['poses'][0, 23] = -30/180*np.array(np.pi)
    big_pose_smpl_param['poses'][0, 26] = 30/180*np.array(np.pi)

    big_pose_xyz, _ = smpl_model(big_pose_smpl_param['poses'], big_pose_smpl_param['shapes'].reshape(-1))
    big_pose_xyz = (np.matmul(big_pose_xyz, big_pose_smpl_param['R'].transpose()) + big_pose_smpl_param['Th']).astype(np.float32)

    # obtain the original bounds for point sampling
    big_pose_min_xyz = np.min(big_pose_xyz, axis=0)
    big_pose_max_xyz = np.max(big_pose_xyz, axis=0)
    big_pose_min_xyz -= 0.05
    big_pose_max_xyz += 0.05
    big_pose_world_bound = np.stack([big_pose_min_xyz, big_pose_max_xyz], axis=0)
    #im_names = sorted(os.listdir(os.path.join(path, 'images')))
    
    if 'w2a' not in path:
        frame_start = 320
        frame_end = 374
        pose_interval = 1
    else:
        frame_start = 0
        frame_end = 10000
        pose_interval = 2

    im_names = [x for x in sorted(os.listdir(os.path.join(path, 'images'))) if int(x[:5]) <= frame_end and int(x[:5]) >= frame_start]
    image_list = [path + "/images/" + x for x in im_names]
    image_list = image_list[::pose_interval]
    print('total length:', len(image_list))
    idx = 0

    gen_msk_root = os.path.join(gen_root, subject, 'gen_masks')
    inpainted_root = os.path.join(gen_root, subject, 'incontext_inpainted')

    #gen_msk_root = None
    for pose_index in range(len(image_list)):
        for view_index in range(len(output_view)):
            if novel_view_vis:
                view_index_look_at = view_index
                view_index = 0

            # Load image, mask, K, D, R, T
            image_path = image_list[pose_index]
            image_name = image_list[pose_index].split('/')[-1].split('.')[0]
            image = np.array(imageio.imread(image_path).astype(np.float32)/255.)

            msk_path = image_path.replace('images', 'masks')
            msk = imageio.imread(msk_path)[:, :, 0]
            msk = (msk != 0).astype(np.uint8)

            if gen_msk_root is not None:
                gen_msk = imageio.imread(os.path.join(gen_msk_root, image_name+'.png'))[:, :]
                gen_msk = (gen_msk != 0).astype(np.uint8)
                if len(gen_msk.shape) == 3:
                    gen_msk = gen_msk[:,:,0]

            if os.path.exists(os.path.join(inpainted_root, image_name.split('/')[-1]+'.png')):
                gen_image = imageio.imread(os.path.join(inpainted_root, image_name.split('/')[-1]+'.png'))
                gen_image = np.array(gen_image.astype(np.float32)/255.)
                gen_image = cv2.resize(gen_image, (image.shape[0], image.shape[1]), interpolation=cv2.INTER_AREA)
                #print(gen_image.shape)
            else:
                gen_image = None

            if not novel_view_vis:
                K = np.array(annots[image_name]['cam_intrinsics'])
                D = np.zeros((5,1))
                #R = np.array(cams['R'][cam_ind])
                #T = np.expand_dims(np.array(annots[image_name]['trans']), axis = 1) / 1000.

                image = cv2.undistort(image, K, D)
                msk = cv2.undistort(msk, K, D)

                if gen_msk_root is not None:
                    gen_msk = cv2.undistort(gen_msk, K, D)
                if gen_image is not None:
                    gen_image = cv2.undistort(gen_image, K, D)

                            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
                w2c = np.array(annots[image_name]['cam_extrinsics'])
            else:
                w2c = np.array(all_cameras[image_name]['extrinsics'])[view_index_look_at]
                K = np.array(all_cameras[image_name]['intrinsics'])[view_index_look_at]
            R = np.transpose(w2c[:3,:3])
            T = w2c[:3, 3]
            image[msk == 0] = 1 if white_background else 0


            # get the world-to-camera transform and set R, T
            

            # Reduce the image resolution by ratio, then remove the back ground
            ratio = image_scaling
            if ratio != 1.:
                H, W = int(image.shape[0] * ratio), int(image.shape[1] * ratio)
                image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)
                msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
                if gen_msk_root is not None:
                    gen_msk = cv2.resize(gen_msk, (W, H), interpolation=cv2.INTER_NEAREST)
                if gen_image is not None:
                    gen_image = cv2.resize(gen_image, (W, H), interpolation=cv2.INTER_AREA)
                K[:2] = K[:2] * ratio

            image = Image.fromarray(np.array(image*255.0, dtype=np.byte), "RGB")

            if gen_image is not None:
                gen_image = Image.fromarray(np.array(gen_image*255.0, dtype=np.byte), "RGB")


            focalX = K[0,0]
            focalY = K[1,1]
            FovX = focal2fov(focalX, image.size[0])
            FovY = focal2fov(focalY, image.size[1])
            #smpl_param_path = os.path.join(path, "smpl_params", '{}.npy'.format(i))
            smpl_param = {}
            Rh = np.expand_dims(np.array(mesh_infos[image_name]['Rh']), axis = 0)
            smpl_param['R'] = np.eye(3)
            smpl_param['Th'] = np.expand_dims(np.array(annots[image_name]['trans']), axis = 0)
            smpl_param['shapes'] = np.expand_dims(np.array(annots[image_name]['betas']), axis = 0)
            smpl_param['poses'] = np.expand_dims(np.array(annots[image_name]['poses']), axis = 0)
            poses = smpl_param['poses']
            poses[:, :3] = Rh
            smpl_param['poses'] = poses
            # load smpl data 
            xyz, _ = smpl_model(poses, smpl_param['shapes'].reshape(-1))
            xyz += np.array(annots[image_name]['trans'])

            converted_image = np.array(image)
            renderer = Renderer(img_size = [converted_image.shape[0], converted_image.shape[1]], cam_intrinsic=K)
            smpl_mesh = trimesh.Trimesh(xyz, smpl_model.faces, process=False)
            R_new  = torch.tensor(w2c[:3,:3])[None].float() # fetch rotatioin matrix from extrinsics
            T_new = torch.tensor(w2c[:3, 3])[None].float()
            rendered_image = render_trimesh(renderer, smpl_mesh, R_new, T_new, 'n')
            if converted_image.shape[0] < converted_image.shape[1]:
                rendered_image = rendered_image[abs(converted_image.shape[0]-converted_image.shape[1])//2:(converted_image.shape[0]+converted_image.shape[1])//2,...] 
            else:
                rendered_image = rendered_image[:,abs(converted_image.shape[0]-converted_image.shape[1])//2:(converted_image.shape[0]+converted_image.shape[1])//2]   

            valid_mask = (rendered_image[:,:,-1] > 0)[:, :, np.newaxis]
        
            output_img = (rendered_image[:,:,:-1] * valid_mask + np.array(image) * (1 - valid_mask)).astype(np.uint8)
  
            # obtain the original bounds for point sampling
            min_xyz = np.min(xyz, axis=0)
            max_xyz = np.max(xyz, axis=0)
            min_xyz -= 0.05
            max_xyz += 0.05
            world_bound = np.stack([min_xyz, max_xyz], axis=0)
            # get bounding mask and bcakground mask
            bound_mask = get_bound_2d_mask(world_bound, K, w2c[:3], image.size[1], image.size[0])

            bound_mask = Image.fromarray(np.array(bound_mask*255.0, dtype=np.byte))

            bkgd_mask = Image.fromarray(np.array(msk*255.0, dtype=np.byte))
            if gen_msk_root is not None:
                gen_msk = Image.fromarray(np.array(gen_msk*255.0, dtype=np.byte))
 

            ##### SMPL to openpose
            # smpl_model_pt
            output = smpl_model_pt(betas = torch.tensor(smpl_param['shapes'].reshape(-1))[None].float(), 
                                   body_pose = torch.tensor(poses)[:,3:].float(), 
                                   global_orient = torch.tensor(poses)[:,:3].float())
            xyz = output.vertices.detach().squeeze().cpu().numpy()
            joints3d = output.joints.detach().squeeze().cpu().numpy()
            joints3d = joints3d[smpl2op]

            joints3d = joints3d + smpl_param['Th']

            
            ###### rotate
            if rotate_smpl is not None:
                w2c = _update_extrinsics(w2c, rotate_smpl, smpl_param['Th'][0])

            #print(np.max(joints3d, axis=0), np.min(joints3d, axis=0))
            joints3d = joints3d @ w2c[:3,:3].T + w2c[:3, [3]].T
            joints2d = joints3d @ K.T
            joints2d = joints2d[:,:2] / joints2d[:,[-1]]
            # #print(np.max(joints2d, axis=0), np.min(joints2d, axis=0))
            canvas = draw_poses(joints2d, image.size[1], image.size[0])
            canvas = Image.fromarray(np.uint8(canvas))
            
            if gen_msk_root is not None:
                gen_msk = gen_msk
            else:
                gen_msk = None
            cam_infos.append(CameraInfo(uid=idx, pose_id=pose_index, R=R, T=T, K=K, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, bkgd_mask=bkgd_mask, 
                            bound_mask=bound_mask, human_mask = bkgd_mask, width=image.size[0], height=image.size[1], 
                            smpl_param=smpl_param, world_vertex=xyz, world_bound=world_bound, 
                            gen_image=gen_image,
                            big_pose_smpl_param=big_pose_smpl_param, big_pose_world_vertex=big_pose_xyz, 
                            big_pose_world_bound=big_pose_world_bound, openpose=canvas, gen_msk=gen_msk))

            idx += 1
            
    return cam_infos

def readOcMotionInfo(path, gen_root, white_background, output_path, eval, rotate_smpl=None):

    train_view = [int(path.split('_')[1])]
    #print(f"train view is {train_view}")
    test_view = [i for i in range(0, 6)]
    test_view.remove(train_view[0])
    test_view = [test_view[0]]

    print("Reading Training Transforms")
    train_cam_infos = readCamerasOcMotion(path, gen_root, train_view, white_background, split='train', rotate_smpl=rotate_smpl)
    #print("Reading Test Transforms")
    #test_cam_infos = readCamerasOcMotion(path, gen_root, test_view, white_background, split='test', novel_view_vis=True, rotate_smpl=rotate_smpl)
    test_cam_infos = []
    # if not eval:
    #     train_cam_infos.extend(test_cam_infos)
    #     test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    if len(train_view) == 1:
        nerf_normalization['radius'] = 1

    ply_path = os.path.join('output', output_path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 6890 #6890
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = train_cam_infos[0].big_pose_world_vertex

        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info



def prepare_smpl_params(smpl_path, pose_index):
    params_ori = dict(np.load(smpl_path, allow_pickle=True))['smpl'].item()
    params = {}
    params['shapes'] = np.array(params_ori['betas']).astype(np.float32)
    params['poses'] = np.zeros((1,72)).astype(np.float32)
    params['poses'][:, :3] = np.array(params_ori['global_orient'][pose_index]).astype(np.float32)
    params['poses'][:, 3:] = np.array(params_ori['body_pose'][pose_index]).astype(np.float32)
    params['R'] = np.eye(3).astype(np.float32)
    params['Th'] = np.array(params_ori['transl'][pose_index:pose_index+1]).astype(np.float32)
    return params

def get_bound_corners(bounds):
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d

def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy

def get_bound_2d_mask(bounds, K, pose, H, W):
    corners_3d = get_bound_corners(bounds)
    corners_2d = project(corners_3d, K, pose)
    corners_2d = np.round(corners_2d).astype(int)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 3, 2, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[4, 5, 7, 6, 4]]], 1) # 4,5,7,6,4
    cv2.fillPoly(mask, [corners_2d[[0, 1, 5, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[2, 3, 7, 6, 2]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 2, 6, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[1, 3, 7, 5, 1]]], 1)
    return mask

def get_mask(path, index, view_index, ims):
    msk_path = os.path.join(path, 'mask_cihp',
                            ims[index][view_index])[:-4] + '.png'
    msk_cihp = imageio.imread(msk_path)
    msk_cihp = (msk_cihp != 0).astype(np.uint8)
    msk = msk_cihp.copy()

    return msk, msk_cihp

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "ZJU_MoCap_refine" : readZJUMoCapRefineInfo,
    "OcMotion" : readOcMotionInfo,
    "MonoCap": readMonoCapdataInfo
}
