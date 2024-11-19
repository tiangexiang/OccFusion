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
os.environ['HF_HOME'] = '/vision/group/occnerf/genhuman/hf_cache/'
os.environ['HF_HUB_CACHE'] = '/vision/group/occnerf/genhuman/hf_cache/'
os.environ['TRANSFORMERS_CACHE'] = '//vision/group/occnerf/genhuman/hf_cache/'
os.environ['GRADIO_TEMP_DIR'] = '/vision/group/occnerf/genhuman/hf_cache/'


import copy
import torch
from random import randint
from utils.loss_utils import l1_loss, l2_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from scene.disc_dataloader import AlphaImageDataset
from utils.general_utils import safe_state
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getProjectionMatrix_refine
import uuid
import imageio
import numpy as np
import cv2
import pickle
from tqdm import tqdm
from utils.image_utils import psnr
from utils.graphics_utils import focal2fov, fov2focal
from argparse import ArgumentParser, Namespace
from matplotlib import pyplot as plt
from arguments import ModelParams, PipelineParams, OptimizationParams
from PIL import Image

    
TENSORBOARD_FOUND = False

import lpips
loss_fn_vgg = lpips.LPIPS(net='vgg').to(torch.device('cuda', torch.cuda.current_device()))

import time
import torch.nn.functional as F

from guidance.op_utils import StableDiffusion
# from guidance.sd_utils import StableDiffusion as OriginalStableDiffusion

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from,
             can_sds_w, pose_sds_w, can_guidance, pose_guidance):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, dataset.smpl_type, dataset.motion_offset_flag, dataset.actor_gender)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    Ll1_loss_for_log = 0.0
    mask_loss_for_log = 0.0
    ssim_loss_for_log = 0.0
    lpips_loss_for_log = 0.0

    pixel_loss_for_log = 0.0
    sds_loss_for_log = 0.0
    pose_sds_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    ##### SDS 
    prompt = 'high contrast to background, solid white silhouette of a person, dressed in normal clothing, legs are straightly stretched, arms are straightly stretched, the person is standing in front of a solid white background, highly realistic, complete body trunk, complete legs, complete feet, complete arms, complete head, complete shoulder, complete hands'
    negative_prompt = 'incomplete body limbs, incomplete body trunk, rotated body, twisted body, ugly, incomplete anatomy, blurry, pixelated obscure, unnatural colors, rich lighting, poor lighting, dull, and unclear, cropped, lowres, low quality, artifacts, duplicate, morbid, mutilated, poorly drawn face, deformed, dehydrated, bad proportions'
    guidance = StableDiffusion('cuda:0')

    guidance.get_text_embeds([prompt], [negative_prompt])

    possible_hors = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, -20, -40, -60, -80, -100, -120, -140, -160]
    possibilities = [1/len(possible_hors)] * len(possible_hors)

    ##### control for SD
    for ph in possible_hors:
        control_image = Image.open('assets/daposesv2/' + '%d.png' % ph)
        guidance.get_control_embeds(control_image, name=str(ph), res=256)

    elapsed_time = 0
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        step_ratio = min(1, (iteration - first_iter) / (opt.iterations - first_iter))

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Start timer
        start_time = time.time()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        if iteration == opt.iterations:
            viewpoint_stack = scene.getTrainCameras().copy()
            for vc in viewpoint_stack:
                if vc.image_name == '00355':
                    viewpoint_cam = vc
                    break
        else:
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        if iteration == first_iter:
            # set up a standard camera for the canonical pose
            # do not change this set up
            canonical_cam = copy.deepcopy(viewpoint_cam)
            K = np.eye(3)
            K[0,0], K[1,1] = 1000., 1000.
            K[0, 2], K[1, 2] = 256, 256
            canonical_cam.projection_matrix =  getProjectionMatrix_refine(torch.Tensor(K).cuda(), 512, 512, 0.001, 1000).transpose(0, 1)

            canonical_cam.image_height = 512
            canonical_cam.image_width = 512
            canonical_cam.FoVx = focal2fov(1000, 512)
            canonical_cam.FoVy = focal2fov(1000, 512)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, alpha, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["render_alpha"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        bkgd_mask = viewpoint_cam.bkgd_mask.cuda()
        if viewpoint_cam.gen_msk is None:
            gen_mask = bkgd_mask.detach().cuda()
        else:
            gen_mask = viewpoint_cam.gen_msk.cuda()
        bound_mask = viewpoint_cam.bound_mask.cuda()
        human_mask = None if viewpoint_cam.human_mask is None else viewpoint_cam.human_mask.cuda()
        alpha_map = np.array(alpha.cpu().detach()).squeeze()
        alpha_mask = alpha_map > 0.01


        Ll1 = l1_loss(image.permute(1,2,0)[bkgd_mask[0]==1], gt_image.permute(1,2,0)[bkgd_mask[0]==1])
        
        mask_loss = l2_loss(alpha[bound_mask==1], gen_mask[bound_mask==1])

        # crop the object region
        x, y, w, h = cv2.boundingRect(bound_mask[0].cpu().numpy().astype(np.uint8))
        img_pred = image[:, y:y + h, x:x + w].unsqueeze(0)
        img_gt = gt_image[:, y:y + h, x:x + w].unsqueeze(0)
        bkgd_mask_vis = bkgd_mask[:, y:y + h, x:x + w]

        #vis_ssim loss
        ssim_map = ssim(img_pred, img_gt, return_all = True)
        ssim_map = ssim_map.reshape([w * h, 3])
        ssim_vis = ssim_map[bkgd_mask_vis.reshape([w*h,]) == 1.0]
        ssim_vis = ssim_vis.mean()


        # lpips loss
        img_pred_restricted = image[:, y:y + h, x:x + w]
        
        img_pred_restricted = img_pred_restricted * bkgd_mask_vis + (1. - bkgd_mask_vis) * torch.ones_like(img_pred_restricted)
        img_gt_restricted = img_gt * bkgd_mask_vis + (1. - bkgd_mask_vis) * torch.ones_like(img_gt)


        lpips_loss = loss_fn_vgg(img_pred_restricted, img_gt_restricted).reshape(-1)

        loss = Ll1 + 2. * mask_loss + 0.1 * (1.0 - ssim_vis) + 0.1 * lpips_loss
        pixel_loss = 10000. * loss

        if np.random.uniform() > 0.25 or iteration == opt.iterations:
            #### pose space sds ####
            openpose = viewpoint_cam.openpose.crop((x, y, x+w, y+h))
            guidance.get_control_embeds(openpose, res=256)

            # gradient separate
            alpha_cropped = alpha[:, y:y + h, x:x + w]
            alpha_cropped_no_grads = alpha_cropped * bkgd_mask_vis # 1,W,H
            alpha_cropped_grads = alpha_cropped * (1. - bkgd_mask_vis)
            alpha_cropped = alpha_cropped_grads + alpha_cropped_no_grads.detach()
            pose_SDS_loss = guidance.train_step(alpha_cropped.unsqueeze(0).expand(-1,3,-1,-1), step_ratio=step_ratio, guidance_scale=pose_guidance, mask=(1. - bkgd_mask_vis), res=256)

            pose_SDS_loss = pose_SDS_loss * pose_sds_w

            loss = pixel_loss + pose_SDS_loss
            loss.backward()
        else:
            #### canonical space sds ####
            loss = pixel_loss
            loss.backward()

            # random hor
            random_hor = np.random.choice(possible_hors, 1, p=possibilities)[0]

            gan_render_pkg = render(canonical_cam, gaussians, pipe, background, canonical=True, hor=random_hor)
            disc_image, disc_alpha, disc_viewspace_point_tensor = gan_render_pkg["render"], gan_render_pkg["render_alpha"], gan_render_pkg["viewspace_points"] # 3, 512, 512

            save_error_name = None
            alpha_mask = (disc_alpha.unsqueeze(0).detach() > 0.01).float() # 1,1,w,h
            SDS_loss = guidance.train_step(disc_alpha.unsqueeze(0).expand(-1,3,-1,-1), step_ratio=step_ratio, guidance_scale=can_guidance, name=str(random_hor), save_error_name=save_error_name, alpha_mask=alpha_mask, res=256)

            SDS_loss = SDS_loss * can_sds_w
            SDS_loss.backward()
        
        # end time
        end_time = time.time()

        # Calculate elapsed time
        elapsed_time += (end_time - start_time)

        if (iteration in testing_iterations):
            print("[Elapsed time]: ", elapsed_time) 

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            pixel_loss_for_log = 0.4 * pixel_loss.item() + 0.6 * pixel_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"#pts": gaussians._xyz.shape[0], "Loss": f"{ema_loss_for_log:.{3}f}", "pixel_Loss": f"{pixel_loss_for_log:.{3}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Start timer
            start_time = time.time()
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            # end time
            end_time = time.time()
            # Calculate elapsed time
            elapsed_time += (end_time - start_time)

            # if (iteration in checkpoint_iterations):
            if (iteration in testing_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def prepare_output_and_logger(args):    
    if not args.model_path:
        args.model_path = os.path.join("./output/", args.exp_name)

        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()

        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : scene.getTrainCameras()})

        smpl_rot = {}
        smpl_rot['train'], smpl_rot['test'] = {}, {}
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0: 
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    smpl_rot[config['name']][viewpoint.pose_id] = {}
                    render_output = renderFunc(viewpoint, scene.gaussians, *renderArgs, return_smpl_rot=True)
                    image = torch.clamp(render_output["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    bound_mask = viewpoint.bound_mask
                    image.permute(1,2,0)[bound_mask[0]==0] = 0 if renderArgs[1].sum().item() == 0 else 1 
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                    lpips_test += loss_fn_vgg(image, gt_image).mean().double()

                    smpl_rot[config['name']][viewpoint.pose_id]['transforms'] = render_output['transforms']
                    smpl_rot[config['name']][viewpoint.pose_id]['translation'] = render_output['translation']

                l1_test /= len(config['cameras']) 
                psnr_test /= len(config['cameras'])   
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])      
                print("\n[ITER {}] Evaluating {} #{}: L1 {} PSNR {} SSIM {} LPIPS {}".format(iteration, config['name'], len(config['cameras']), l1_test, psnr_test, ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)

        # Store data (serialize)
        save_path = os.path.join(scene.model_path, 'smpl_rot', f'iteration_{iteration}')

        os.makedirs(save_path, exist_ok=True)
        with open(save_path+"/smpl_rot.pickle", 'wb') as handle:
            pickle.dump(smpl_rot, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1_200, 2_000, 3_000, 7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1_200, 2_000, 3_000, 7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)


    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, 20., 20., 0, 0)
    # All done
    print("Optimization Stage complete.")
