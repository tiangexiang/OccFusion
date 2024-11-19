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
import time
import torch.nn.functional as F

import lpips
loss_fn_vgg = lpips.LPIPS(net='vgg').to(torch.device('cuda', torch.cuda.current_device()))

TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from,
             mask_scale, l1_scale, gen_lpips_scale, gen_l2_scale):
    first_iter = int(checkpoint.split('/')[-1][6:-4])
    opt.iterations = opt.iterations + first_iter
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, dataset.smpl_type, dataset.motion_offset_flag, dataset.actor_gender)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    if checkpoint:
        print('LOAD from CHECKPOINT:', checkpoint)
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

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, alpha, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["render_alpha"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda().clone()
        bkgd_mask = viewpoint_cam.bkgd_mask.cuda().clone()
        gen_mask = viewpoint_cam.gen_msk.cuda().clone()
        bound_mask = viewpoint_cam.bound_mask.cuda().clone()

       #print(bound_mask.shape, torch.unique(bound_mask), torch.max(bound_mask))


        alpha_map = np.array(alpha.cpu().detach()).squeeze()
        invisible_mask = gen_mask * (1. - bkgd_mask)
        bound_vis_mask = torch.logical_and(bound_mask==1, (invisible_mask).long()==0)[0]


        Ll1 = l1_loss(image.permute(1,2,0)[bound_vis_mask==1], gt_image.permute(1,2,0)[bound_vis_mask==1]) * l1_scale
        mask_loss = l2_loss(alpha[bound_mask==1], gen_mask[bound_mask==1]) * mask_scale #* 0.1


        # crop the object region
        x, y, w, h = cv2.boundingRect(bound_mask[0].cpu().numpy().astype(np.uint8))
        img_pred = image[:, y:y + h, x:x + w].unsqueeze(0)
        img_gt = gt_image[:, y:y + h, x:x + w].unsqueeze(0)
        bkgd_mask_vis = bkgd_mask[:, y:y + h, x:x + w]
        gen_mask = gen_mask[:, y:y + h, x:x + w]

        # lpips loss
        img_pred_restricted = image[:, y:y + h, x:x + w]

        img_pred_restricted = img_pred_restricted * bkgd_mask_vis + (1. - bkgd_mask_vis) * torch.ones_like(img_pred_restricted)
        img_gt_restricted = img_gt * bkgd_mask_vis + (1. - bkgd_mask_vis) * torch.ones_like(img_gt)

        loss = Ll1 + mask_loss
        pixel_loss = loss


        ###### stage2 generated images
        if viewpoint_cam.gen_image is None:
            gen_image = viewpoint_cam.original_image.cuda().clone()
        else:
            gen_image = viewpoint_cam.gen_image.cuda().clone()

        gen_image = gen_image[:, y:y + h, x:x + w]
        mask_for_gen = (alpha > 0.9).float()
        mask_for_gen = mask_for_gen[:, y:y + h, x:x + w]
        rgb_image = image[:, y:y + h, x:x + w]
        gen_image = gen_image * gen_mask + (1. - gen_mask) * torch.ones_like(gen_image)

        lpips_loss = loss_fn_vgg(rgb_image, gen_image).reshape(-1) * gen_lpips_scale
        invisible_mask = mask_for_gen.long().squeeze()

        gen_loss = l1_loss(rgb_image, gen_image) * gen_l2_scale 

        pixel_loss = pixel_loss + lpips_loss + gen_loss
        pixel_loss.backward()


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

            ###### disable saving during search
            if (iteration in [opt.iterations]):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Start timer
            start_time = time.time()
            # Densification
            opt.densification_interval = 200
            if iteration < opt.densify_until_iter + first_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    prune_only = False
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, kl_threshold=0.4, t_vertices=viewpoint_cam.big_pose_world_vertex, iter=iteration, prune_only=prune_only)
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            # end time
            end_time = time.time()
            # Calculate elapsed time
            elapsed_time += (end_time - start_time)

            ###### disable saving during search
            if (iteration in [opt.iterations]):
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
    parser.add_argument("--preset", type=int, default = 0)


    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    #network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
        
    mask_scale, l1_scale, gen_lpips, gen_l2 = 0.1, 2., 0.2, 0.1

    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, mask_scale, l1_scale, gen_lpips, gen_l2)
    # All done
    print("Refinement Stage complete.")
