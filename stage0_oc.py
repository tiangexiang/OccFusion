import os
import cv2
from utils.preprocessing_utils import (smpl_to_pose, PerspectiveCamera, Renderer, render_trimesh_depth, render_trimesh, \
                                estimate_translation_cv2, transform_smpl)
from utils.pose_utils import draw_poses

import json
from matplotlib import pyplot as plt

from PIL import Image
from diffusers import DPMSolverMultistepScheduler, LMSDiscreteScheduler
from diffusers import StableDiffusionXLControlNetInpaintPipeline, StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, StableDiffusionControlNetInpaintPipeline
import torch
import numpy as np
import trimesh
import cv2
import argparse

from segment_anything_hq import sam_model_registry, SamPredictor

# from smpl.smpl_numpy import SMPL
from smplx import SMPL



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
        default=30,
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

    # Parse the arguments
    args = parser.parse_args()



    smpl2op = [24,12,17,19,21,16,18,20,2,5,8,1,4,7,25,26,27,28]


    ###### load controlnet and SD, can play with the checkpoints, maybe one is better than the others
    checkpoint = "lllyasviel/sd-controlnet-openpose"
    #checkpoint = 'fusing/stable-diffusion-v1-5-controlnet-openpose'
    #checkpoint = "lllyasviel/control_v11p_sd15_openpose"

    controlnet = ControlNetModel.from_pretrained(
        checkpoint , torch_dtype=torch.float16
    )


    checkpoint = "runwayml/stable-diffusion-inpainting"
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        checkpoint, controlnet=controlnet, torch_dtype=torch.float16, variant="fp16", safety_checker = None,
        requires_safety_checker = False
    ).to('cuda:0')

    # pipe.enable_model_cpu_offload()
    # pipe.enable_xformers_memory_efficient_attention()
    #pipe.enable_vae_slicing()
    #pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    #pipe.scheduler = LMSDiscreteScheduler(pipe.scheduler.config)



    ######## define prompts here
    prompt = 'clean background, high contrast to background, a person only, plain clothes, simple clothes, natural body, natural limbs, no texts, no overlay' + ', two arms, two legs, one head'
    negative_prompt = 'multiple objects, occlusions, complex pattern, fancy clothes, longbody, lowres, bad anatomy, bad hands, bad feet, missing fingers, cropped, worst quality, low quality, blurry'
    guess_mode = True



    ######## define SMPL
    smpl_model = SMPL(gender='neutral', model_path='assets/SMPL_NEUTRAL.pkl')


    ######## define SAM
    sam_checkpoint = args.sam_checkpoint 
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)


    ######## load ocmotion annotation
    seq = args.subject # e.g. '0011_02_1_w2a'
    path = args.data_root #os.path.join(args.data_root, seq)

    json_file = os.path.join(path, 'metadata.json')
    with open(json_file) as f:
        annots = json.load(f)
    f.close()

    frame_start = 0
    frame_end = 100000
    interval = 2


    im_names = [x for x in sorted(os.listdir(os.path.join(path, 'images'))) if int(x[:5]) <= frame_end and int(x[:5]) >= frame_start]

    image_list = [path + "/images/" + x for x in im_names][::interval]
    mask_list = [path + "/masks/" + x for x in im_names][::interval]


    ###### where to save generations?
    stage0_generation_path = f'oc_generations/{seq}/generations'
    stage0_generation_mask_path = f"oc_generations/{seq}/gen_masks"
    #stage0_generation_candidate_path = f'oc_generations/{seq}/generations_candidates'
    stage0_mask_path = f'oc_generations/{seq}/masks'


    if not os.path.exists(f'oc_generations/'):
        os.mkdir('oc_generations/')

    if not os.path.exists(f'oc_generations/{seq}'):
        os.mkdir(f'oc_generations/{seq}')

    if not os.path.exists(f'{stage0_generation_path}'):
        os.mkdir(f'{stage0_generation_path}')

    if not os.path.exists(f'{stage0_generation_mask_path}'):
        os.mkdir(f'{stage0_generation_mask_path}')

    # if not os.path.exists(f'{stage0_generation_candidate_path}'):
    #     os.mkdir(f'{stage0_generation_candidate_path}')

    # if not os.path.exists(f'{stage0_mask_path}'):
    #     os.mkdir(f'{stage0_mask_path}')

    


    for index in range(len(image_list)):
        
        image_path = image_list[index]
        mask_path = mask_list[index]
        image_name = image_list[index].split('/')[-1].split('.')[0]
        print('start', image_name)

        if not args.overwrite and os.path.exists(f'{stage0_generation_path}/{image_name}.png'):
            continue

        # if os.path.exists(f'{stage0_generation_candidate_path}/{image_name}'):
        #     continue


        ####### load original image and SAM mask
        input_img = cv2.imread(image_path)
        vis_mask = cv2.imread(mask_path)[:,:,[0]] / 255.

        ####### preprocess image to white background
        image = input_img #[:, :, ::-1]
        print('original size', image.shape)
        image = image * vis_mask + np.ones_like(image) * (1.-vis_mask) * 255.
        image = np.uint8(image)

        ####### fetech metadata from ocmotion annotation
        K = np.array(annots[image_name]['cam_intrinsics'])
        w2c = np.array(annots[image_name]['cam_extrinsics'])

        width = input_img.shape[0]
        height = input_img.shape[1]

        renderer = Renderer(img_size = [width, height], cam_intrinsic=K)

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

        smpl_mesh = trimesh.Trimesh(xyz, smpl_model.faces, process=False)
        R = torch.tensor(w2c[:3,:3])[None].float()
        T = torch.tensor(w2c[:3, 3])[None].float() 
        rendered_image = render_trimesh(renderer, smpl_mesh, R, T, 'n')
        smpl_depth = render_trimesh_depth(renderer, smpl_mesh, R, T, 'n')[0]
        smpl_depth = smpl_depth[abs(input_img.shape[0]-input_img.shape[1])//2:(input_img.shape[0]+input_img.shape[1])//2,:,0].cpu().numpy()
        rendered_image = rendered_image[abs(input_img.shape[0]-input_img.shape[1])//2:(input_img.shape[0]+input_img.shape[1])//2,...] 
        valid_mask = (rendered_image[:,:,-1] > 0).astype(np.float32)[:, :, np.newaxis]
        output_img = (rendered_image[:,:,:-1] * valid_mask + input_img * (1 - valid_mask)).astype(np.uint8)

        ####### decide ROI, will crop image to keep only the part that has human
        x, y, w, h = cv2.boundingRect(valid_mask[:,:,0].astype(np.uint8))
        x = max(0, x - 20) # add padding at the boundary
        y = max(0, y - 20) # add padding at the boundary
        w = w + (8 - (w+40) % 8) + 40
        x2 = x + w
        x2 = min(x2, vis_mask.shape[1] - 1)

        h = h + (8 - (h+40) % 8)  + 40
        y2 = y + h
        y2 = min(y2, vis_mask.shape[0] - 1)

        ####### keep track of the cropped region with respect to the original image
        # save_mask = np.zeros_like(vis_mask)
        # save_mask[y:y2, x:x2, 0] += 255
        # cv2.imwrite(f'{stage0_mask_path}/{image_name}.png', save_mask)

        
        ###############################################################
        ####### which part of the image do we need to inpaint/generate?
        ###############################################################

        ####### reverse mask of the visibile SAM mask
        dialted_smpl_mask = np.uint8((1. - vis_mask) * 255)

        ####### make inpaint mask to be passed to stable diffusion (only the masked region will be inpainted)
        dialted_smpl_mask = np.concatenate([dialted_smpl_mask] * 3, axis=-1)
        dialted_smpl_mask = dialted_smpl_mask[y:y2, x:x2]

        mask = Image.fromarray(dialted_smpl_mask)

        ####### crop ROI
        image = image[y:y2, x:x2, ::-1]
        vis_mask = vis_mask[y:y2, x:x2]
        
        ####### make image
        image = Image.fromarray(np.uint8(image))
        

        ####### start to make openpose canvas for using control net. First, project 3d kepoints to 2d image
        joints3d = joints3d @ w2c[:3,:3].T + w2c[:3, [3]].T
        joints2d = joints3d @ K.T
        joints2d = joints2d[:,:2] / joints2d[:,[-1]]


        ####### remove self-occluded joints via z-buffer test (Sec. 4.1)
        for jidx, j2d in enumerate(joints2d):
            if abs(smpl_depth[int(j2d[1]), int(j2d[0])] - joints3d[jidx,-1]) > 0.3: # 0.3 is the hyper parameter
                joints2d[jidx] *= -1


        ####### making openpose canvas, this is the openpose control to control net
        canvas = draw_poses(joints2d, input_img.shape[0], input_img.shape[1])
        canvas = canvas[y:y2, x:x2]
        canvas = Image.fromarray(np.uint8(canvas))


        ####### generate only one sample
        img = pipe(prompt=prompt, image=image, mask_image=mask,  control_image=canvas, negative_prompt=negative_prompt, num_inference_steps=args.num_inference_steps, controlnet_conditioning_scale=1.0).images[0]
        img = np.array(img)
        
        inpainted_rgb = np.ones_like(input_img) * 255
        inpainted_rgb[y:y2, x:x2] = img[:,:,::-1]
        inpainted_rgb = inpainted_rgb.astype(np.uint8)
        if args.save_generation:
            cv2.imwrite(f'{stage0_generation_path}/{image_name}.png', inpainted_rgb)


        ####### run SAM to get mask
        # predictor.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        predictor.set_image(img)

        input_label = np.ones(joints2d.shape[0])

        joints2d[:,0] -= x
        joints2d[:,1] -= y


        masks, _, _ = predictor.predict(
                    point_coords=joints2d,
                    point_labels=input_label,
                    box = None,
                    multimask_output=False,
                    hq_token_only=False, 
                )

        masks = masks.squeeze()
        
        save_mask = np.zeros_like(input_img)[:,:,0]
        save_mask[y:y2, x:x2] = masks * 255
        
        cv2.imwrite(f'{stage0_generation_mask_path}/{image_name}.png', save_mask)

    print("Initialization Stage complete.")

        