import torch
from scene import Scene
from scene.cameras import orbit_camera
import os
import time
import pickle
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
from utils.image_utils import psnr
from utils.loss_utils import ssim


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, canonical=False, canonical_pose=False):
    render_path = os.path.join(model_path, name, "{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    rgbs = []
    rgbs_gt = []
    alphas = []
    elapsed_time = 0

    hors = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, -20, -40, -60, -80, -100, -120, -140, -160]
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        gt = view.original_image[0:3, :, :].cuda()


        frame_name = view.image_name.split('/')[-1]

        bound_mask = view.bound_mask

        # Start timer
        start_time = time.time() 

        render_output = render(view, gaussians, pipeline, background)

        rendering = render_output["render"]
        alpha = render_output["render_alpha"]
        
        # end time
        end_time = time.time()

        # Calculate elapsed time
        elapsed_time += end_time - start_time

        rendering.permute(1,2,0)

        rgbs.append(rendering)
        rgbs_gt.append(gt)
        alphas.append(alpha)

    # Calculate elapsed time
    print("Elapsed time: ", elapsed_time, " FPS: ", len(views)/elapsed_time) 

    for id in range(len(views)):
        rendering = rgbs[id]
        gt = rgbs_gt[id]
        rendering = torch.clamp(rendering, 0.0, 1.0)
        gt = torch.clamp(gt, 0.0, 1.0)
        alpha = torch.clamp(alphas[id], 0.0, 1.0)

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(id) + ".png"))
        torchvision.utils.save_image(alpha, os.path.join(render_path, '{0:05d}'.format(id) + "_alpha.png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(id) + ".png"))


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, canonical : bool, canonical_pose : bool):
    with torch.no_grad():

        gaussians = GaussianModel(dataset.sh_degree, dataset.smpl_type, dataset.motion_offset_flag, dataset.actor_gender)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
   

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, canonical, canonical_pose)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, canonical, canonical_pose)




if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--canonical", action="store_true")
    parser.add_argument("--canonical_pose", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.canonical, args.canonical_pose)