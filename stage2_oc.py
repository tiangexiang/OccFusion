import os
os.environ['HF_HOME'] = '/vision/group/occnerf/genhuman/hf_cache/'
os.environ['HF_HUB_CACHE'] = '/vision/group/occnerf/genhuman/hf_cache/'
os.environ['TRANSFORMERS_CACHE'] = '//vision/group/occnerf/genhuman/hf_cache/'
os.environ['GRADIO_TEMP_DIR'] = '/vision/group/occnerf/genhuman/hf_cache/'
import cv2
import sys
sys.path.insert(0, '/vision/u/xtiange/wildnerf/vid2avatar/preprocessing/')
sys.path.insert(0, '/vision/u/xtiange/genhuman/gen-human/')
from preprocessing_utils import (smpl_to_pose, PerspectiveCamera, Renderer, render_trimesh_depth, render_trimesh, \
                                estimate_translation_cv2, transform_smpl)
from utils.pose_utils import draw_poses
from scene.cameras import orbit_camera

import json
from matplotlib import pyplot as plt

from PIL import Image
from diffusers import StableDiffusionXLControlNetInpaintPipeline, StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, StableDiffusionControlNetInpaintPipeline
import torch
import numpy as np
from controlnet_aux import OpenposeDetector
from diffusers.utils import load_image
import trimesh
import cv2
import argparse

# from smpl.smpl_numpy import SMPL
from smplx import SMPL


smpl2op = [24,12,17,19,21,16,18,20,2,5,8,1,4,7,25,26,27,28]


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Example script demonstrating argparse usage."
    )

    # Positional argument (required)
    parser.add_argument(
        "--data_root", 
        type=str, 
        help="Path to the datasets"
    )

    parser.add_argument(
        "--subject", 
        type=str, 
        help="subject"
    )

    parser.add_argument(
        "--sam_checkpoint", 
        type=str, 
        help="sam checkpoint"
    )

    parser.add_argument(
        "--num_inference_steps", 
        type=int, 
        default=35,
        help="inference steps for SD"
    )

    parser.add_argument(
        "--overwrite", 
        action="store_true",
        help="overwrite generations?"
    )

    parser.add_argument(
        "--save_generation", 
        action="store_true",
        help="save inpainted RGB?"
    )

    parser.add_argument(
        "--stage1_render_path", 
        type=str, 
        help="renderings from the optimization stage"
    )

    

    # Parse the arguments
    args = parser.parse_args()

    op_controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-openpose" , torch_dtype=torch.float16
    )


    checkpoint = "runwayml/stable-diffusion-inpainting"
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        checkpoint, controlnet=op_controlnet, torch_dtype=torch.float16, variant="fp16", safety_checker = None,
        requires_safety_checker = False
    ).to('cuda:0')
    #pipe.enable_model_cpu_offload()
    #pipe.enable_vae_slicing()
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)


    ######## define prompts here
    prompt = 'the same person standing in two different rooms' + 'realistic, typical, natural body, two arms, two legs, real person' # , dark gray shorts, black shirt, clear skin'
    negative_prompt = 'holding something, complex textures, malformed, abnormal, unnatural, cartoonish, occlusions, low quality, objects, blurry'
    guess_mode = True 

    smpl_model = SMPL(gender='neutral', model_path='assets/SMPL_NEUTRAL.pkl')


    ######## load ocmotion annotation
    subject = args.subject
    path = args.data_root

    json_file = os.path.join(path, 'metadata.json')
    with open(json_file) as f:
        annots = json.load(f)
    f.close()

    pose_interval = 2

    no_occ_path = os.path.join(path, 'images')
    im_names = [x for x in sorted(os.listdir(no_occ_path))]

    image_list = [no_occ_path + '/' + x for x in im_names]
    vis_mask_list = [os.path.join(path, 'masks') + '/' + x for x in im_names]


    stage1_render_path = args.stage1_render_path 

    stage2_inpaint_path = f'oc_generations/{subject}/incontext_inpainted/'

    if not os.path.exists(f'oc_generations/{subject}'):
        os.mkdir(f'oc_generations/{subject}')

    if not os.path.exists(f'{stage2_inpaint_path}'):
        os.mkdir(f'{stage2_inpaint_path}')


    alpha_threshold = 0.65

    counter = 0
    for count_index, index in enumerate(range(len(image_list))[::pose_interval]):
        
        image_path = image_list[index]
        vis_mask_path = vis_mask_list[index]

        image_name = image_list[index].split('/')[-1][:-4]


        alpha_mask_original = cv2.imread(os.path.join(stage1_render_path, '%05d_alpha.png' % (count_index))) / 255.
        alpha_mask = (alpha_mask_original > alpha_threshold).astype(np.float32)

        x, y, w, h = cv2.boundingRect(alpha_mask[:,:,0].astype(np.uint8) * 255)
        x = max(0, x - 20) # add padding at the boundary
        y = max(0, y - 20) # add padding at the boundary
        w = w + (8 - (w+40) % 8) + 40
        x2 = x + w
        x2 = min(x2, alpha_mask.shape[1] - 1)
        h = h + (8 - (h+40) % 8)  + 40
        y2 = y + h
        y2 = min(y2, alpha_mask.shape[0] - 1)

        alpha_mask = alpha_mask[y:y2, x:x2]

        stage1_render = cv2.imread(os.path.join(stage1_render_path, '%05d.png' % (count_index)))[:,:,::-1]
        save_img = np.ones_like(stage1_render) * 255
        stage1_render = stage1_render[y:y2, x:x2]

        print('start', image_name)

        ####### load original image and SAM mask
        input_img = cv2.imread(image_path)[:,:,::-1]
        original_size = input_img.shape
        input_img = cv2.resize(input_img, (input_img.shape[1] // 2, input_img.shape[0] // 2))
        
        input_img = input_img[y:y2, x:x2]

        vis_mask = cv2.imread(vis_mask_path)[:,:,[0]] / 255. # np.ones_like(alpha_mask).astype(np.float32)
        vis_mask = cv2.resize(vis_mask, (vis_mask.shape[1] // 2, vis_mask.shape[0] // 2))
        if len(vis_mask.shape) < 3:
            vis_mask = vis_mask[:,:,None]

        vis_mask = vis_mask[y:y2, x:x2]
        invisible_mask = alpha_mask * (1. - vis_mask)
        invisible_mask = np.uint8(invisible_mask * 255.)

        invisible_mask = np.hstack((np.zeros_like(invisible_mask), invisible_mask))
        invisible_mask = Image.fromarray(invisible_mask)

        ####### preprocess image to white background
        image = input_img
        image = image * vis_mask + np.ones_like(image) * (1.-vis_mask) * 255.
        image = np.uint8(image)

        image = np.hstack((stage1_render, image))
        image = Image.fromarray(np.uint8(image))


        smpl_param_path = os.path.join(path, "smpl_params", '{}.npy'.format(int(image_name)))

        K = np.array(annots[image_name]['cam_intrinsics'])
        w2c = np.array(annots[image_name]['cam_extrinsics'])
        pose = np.expand_dims(np.array(annots[image_name]['poses']), axis = 0)
        shape = np.expand_dims(np.array(annots[image_name]['betas']), axis = 0).reshape(-1)
        trans = np.expand_dims(np.array(annots[image_name]['trans']), axis = 0)


        ####### render SMPL mesh
        output = smpl_model(betas = torch.tensor(shape)[None].float(), body_pose = torch.tensor(pose)[:,3:].float(), global_orient = torch.tensor(pose)[:,:3].float())
        xyz = output.vertices.detach().squeeze().cpu().numpy()
        joints3d = output.joints.detach().squeeze().cpu().numpy()
        joints3d = joints3d[smpl2op]
        xyz = xyz + trans
        joints3d = joints3d + trans


        # ####### start to make openpose canvas for using control net, first, project 3d kepoints to 2d image
        joints3d = joints3d @ w2c[:3,:3].T + w2c[:3, [3]].T
        joints2d = joints3d @ K.T
        joints2d = joints2d[:,:2] / joints2d[:,[-1]]


        vis_mask = (alpha_mask_original > alpha_threshold).astype(np.float32)
        vis_mask[vis_mask[:,:,0] > alpha_threshold,:] = np.array([150, 5, 61])
        vis_mask = np.uint8(vis_mask)
        vis_mask = vis_mask[y:y2, x:x2]


        ####### making openpose canvas, this is the openpose control to control net
        op_canvas = draw_poses(joints2d, original_size[0], original_size[1])
        op_canvas = cv2.resize(op_canvas, (op_canvas.shape[1]//2, op_canvas.shape[0]//2))
        op_canvas = op_canvas[y:y2, x:x2]
        

        ####### in-context inpainting (Sec. 4.3) #######
        op_canvas = np.hstack((op_canvas, op_canvas))

        op_canvas = Image.fromarray(np.uint8(op_canvas))

        vis_mask = np.hstack((vis_mask, vis_mask))
        canvas = Image.fromarray(np.uint8(vis_mask))

        #### resizing
        width, height = image.size
        owdith, oheight = width, height
        new_width = (width // 8) * 8
        new_height = (height // 8) * 8
        new_width = min(new_width, width)
        new_height = min(new_height, height)
        
        resized_image = image.resize((new_width, new_height))
        resized_invisible_mask = invisible_mask.resize((new_width, new_height))
        resized_canvas = canvas.resize((new_width, new_height))
        resized_op_canvas = op_canvas.resize((new_width, new_height))


        ####### run only one generation
        img = pipe(prompt=prompt, image=resized_image, mask_image=resized_invisible_mask,  control_image=resized_op_canvas, negative_prompt=negative_prompt, num_inference_steps=args.num_inference_steps, controlnet_conditioning_scale=0.3, guess_mode=guess_mode).images[0]

        img = img.resize((owdith, oheight))
        img = img.crop((img.size[0]//2,0,img.size[0],img.size[1]))
        img = np.array(img)
        save_img[y:y2, x:x2] = img
        img = Image.fromarray(save_img)
    
        img.save(f'{stage2_inpaint_path}/{image_name}.png')

        counter += 1
        # if counter == 50:
        #     break

     # All done
    print("Incontext inpainting complete.")
