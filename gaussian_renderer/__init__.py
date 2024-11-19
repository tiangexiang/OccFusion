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

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from scene.cameras import orbit_camera
from utils.sh_utils import eval_sh
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getProjectionMatrix_refine
import numpy as np


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, return_smpl_rot=False, transforms=None, translation=None, canonical=False, canonical_pose=False, hor=0):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    if canonical or canonical_pose:
        R = np.eye(3)
        T = np.zeros_like(viewpoint_camera.T)
        c2w = orbit_camera(0, hor, 4.5) # front view
        w2c = np.linalg.inv(c2w)
        # rectify...
        
        w2c[1:3, :3] *= -1
        w2c[:3, 3] *= -1
        #print(hor)
        viewpoint_camera.world_view_transform = torch.tensor(w2c).transpose(0, 1).cuda()
        viewpoint_camera.full_proj_transform = viewpoint_camera.world_view_transform @ viewpoint_camera.projection_matrix
        viewpoint_camera.camera_center = -torch.tensor(c2w[:3, 3]).cuda()

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform, 
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    means3D = pc.get_xyz

    if not pc.motion_offset_flag:
        if canonical:
            _, means3D, _, transforms, _ = pc.coarse_deform_c2source(means3D[None], viewpoint_camera.big_pose_smpl_param,
                viewpoint_camera.big_pose_smpl_param,
                viewpoint_camera.big_pose_world_vertex[None])
        else:
            _, means3D, _, transforms, _ = pc.coarse_deform_c2source(means3D[None], viewpoint_camera.smpl_param,
                viewpoint_camera.big_pose_smpl_param,
                viewpoint_camera.big_pose_world_vertex[None])
    else:
        if transforms is None:
            # pose offset
            dst_posevec = viewpoint_camera.smpl_param['poses'][:, 3:]
            pose_out = pc.pose_decoder(dst_posevec)
            correct_Rs = pose_out['Rs']

            # SMPL lbs weights
            lbs_weights = pc.lweight_offset_decoder(means3D[None].detach())
            lbs_weights = lbs_weights.permute(0,2,1)

            # transform points
            if canonical:
                transforms = None
                ### center means3d
                means3D = means3D - torch.mean(means3D, dim=0, keepdim=True)

            elif canonical_pose:
                _, means3D, _, _, _ = pc.coarse_deform_c2source(means3D[None], viewpoint_camera.smpl_param,
                    viewpoint_camera.big_pose_smpl_param,
                    viewpoint_camera.big_pose_world_vertex[None], lbs_weights=lbs_weights, correct_Rs=correct_Rs, return_transl=return_smpl_rot)
                means3D = means3D - torch.mean(means3D, dim=0, keepdim=True).detach()
            else:
                _, means3D, _, transforms, translation = pc.coarse_deform_c2source(means3D[None], viewpoint_camera.smpl_param,
                    viewpoint_camera.big_pose_smpl_param,
                    viewpoint_camera.big_pose_world_vertex[None], lbs_weights=lbs_weights, correct_Rs=correct_Rs, return_transl=return_smpl_rot)
                
        else:
            correct_Rs = None
            means3D = torch.matmul(transforms, means3D[..., None]).squeeze(-1) + translation

    
    means3D = means3D.squeeze()

    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        if transforms is not None:
            cov3D_precomp = pc.get_covariance(scaling_modifier, transforms.squeeze())
        else:
            cov3D_precomp = pc.get_covariance(scaling_modifier, transforms)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (means3D - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    #print(means3D.shape, torch.mean(means3D,dim=0))
    rendered_image, radii, depth, alpha = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    # print(radii.shape, depth.shape, torch.sum(radii > 0))
    # print(radii)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "render_depth": depth,
            "render_alpha": alpha,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "transforms": transforms,
            "translation": translation,
            "correct_Rs": correct_Rs,}
