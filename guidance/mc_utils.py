from diffusers import (
    DDIMScheduler,
    StableDiffusionPipeline,
)
from diffusers.utils.import_utils import is_xformers_available
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, StableDiffusionControlNetInpaintPipeline

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from matplotlib import pyplot as plt

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


class StableDiffusion(nn.Module):
    def __init__(
        self,
        device,
        fp16=True,
        vram_O=False,
        sd_version="2.1",
        hf_key=None,
        t_range=[0.02, 0.98],
    ):
        super().__init__()

        self.device = device
        self.sd_version = sd_version

        if hf_key is not None:
            print(f"[INFO] using hugging face custom model key: {hf_key}")
            model_key = hf_key
        elif self.sd_version == "2.1":
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == "2.0":
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == "1.5":
            model_key = "runwayml/stable-diffusion-v1-5"
        else:
            raise ValueError(
                f"Stable-diffusion version {self.sd_version} not supported."
            )

        #checkpoint = "lllyasviel/sd-controlnet-openpose"
        checkpoint = "lllyasviel/control_v11p_sd15_openpose"
        controlnet_op = ControlNetModel.from_pretrained(
            checkpoint , torch_dtype=torch.float16
        )

        controlnet_mask = ControlNetModel.from_pretrained(
            'lllyasviel/control_v11p_sd15_seg', torch_dtype=torch.float16
        )

        self.dtype = torch.float16 if fp16 else torch.float32

        # Create model
        # pipe = StableDiffusionPipeline.from_pretrained(
        #     model_key, torch_dtype=self.dtype
        # )
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=[controlnet_op, controlnet_mask], torch_dtype=torch.float16, variant="fp16"
        ).to('cuda:0')

        if vram_O:
            pipe.enable_sequential_cpu_offload()
            pipe.enable_vae_slicing()
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing(1)
            # pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.controlnet = pipe.controlnet
        self.prepare_image = pipe.prepare_image

        self.scheduler = DDIMScheduler.from_pretrained(
            model_key, subfolder="scheduler", torch_dtype=self.dtype
        )

        del pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience

        self.embeddings = {}

        self.control_down = {}
        self.control_mid = {}
        self.control_image = {}


    @torch.no_grad()
    def ratio_preserve_resize(self, x, res=256):
        if len(x.shape) < 4:
            x = x.unsqueeze(0)

        if x.shape[-2] >= x.shape[-1]:
            ratio = x.shape[-2] / res
            x = F.interpolate(x, (res, int(x.shape[-1] / ratio)), mode='bilinear', align_corners=False)
            #longsize_rate = 256 / x.shape[-2]
            #print(x.shape)
            short_size = x.shape[-1]
            #x = F.interpolate(x, (256, short_size), mode='bilinear', align_corners=False)
            x = F.pad(x, ((res-short_size) // 2, res - ((res-short_size) // 2 + short_size), 0, 0), mode='replicate')
            
            #print(x.shape, '??')
        else:
            ratio = x.shape[-1] / res
            x = F.interpolate(x, (int(x.shape[-2] / ratio), res), mode='bilinear', align_corners=False)
            #longsize_rate = 256 / x.shape[-1]
            short_size = x.shape[0]#int(longsize_rate *  x.shape[-2])
            #x = F.interpolate(x, (short_size, 256), mode='bilinear', align_corners=False)
            x = F.pad(x, (0, 0, (res-short_size) // 2, res - ((res-short_size) // 2 + short_size)), mode='replicate')
        return x

    @torch.no_grad()
    def get_text_embeds(self, prompts, negative_prompts):
        pos_embeds = self.encode_text(prompts)  # [1, 77, 768]
        neg_embeds = self.encode_text(negative_prompts)
        self.embeddings['pos'] = pos_embeds
        self.embeddings['neg'] = neg_embeds

        # directional embeddings
        for d in ['front', 'side', 'back']:
            embeds = self.encode_text([f'{p}, {d} view' for p in prompts])
            self.embeddings[d] = embeds

    @torch.no_grad()
    def get_control_embeds_op(self, image, name='op', res=512):
        vis_mask = self.ratio_preserve_resize(image, res)[0]
        vis_mask = image.permute(1,2,0).detach().cpu().numpy()
        image = image.resize((res,res))
        control_image = self.prepare_image(
                image=image,
                width=image.size[0],
                height=image.size[1],
                batch_size=1,
                num_images_per_prompt=1,
                device=self.device,
                dtype=self.controlnet.dtype,
                do_classifier_free_guidance=False,
                guess_mode=False,
            )

        self.control_image[name] = control_image

    @torch.no_grad()
    def get_control_embeds_mask(self, image, name='mask', res=512):
        vis_mask = self.ratio_preserve_resize(image, res)[0]
        vis_mask = image.permute(1,2,0).detach().cpu().numpy()
        vis_mask[vis_mask[:,:,0] > 0.5,:] = np.array([150, 5, 61])
        #print(vis_mask.shape)
        image = Image.fromarray(np.uint8(vis_mask))
        image = image.resize((res,res))
        
        control_image = self.prepare_image(
                image=image,
                width=image.size[0],
                height=image.size[1],
                batch_size=1,
                num_images_per_prompt=1,
                device=self.device,
                dtype=self.controlnet.dtype,
                do_classifier_free_guidance=False,
                guess_mode=False,
            )

        self.control_image[name] = control_image

    def encode_text(self, prompt):
        # prompt: [str]
        inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]
        return embeddings

    @torch.no_grad()
    def refine(self, pred_rgb,
               guidance_scale=100, steps=50, strength=0.8,
        ):

        batch_size = pred_rgb.shape[0]
        pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
        latents = self.encode_imgs(pred_rgb_512.to(self.dtype))
        # latents = torch.randn((1, 4, 64, 64), device=self.device, dtype=self.dtype)

        self.scheduler.set_timesteps(steps)
        init_step = int(steps * strength)
        latents = self.scheduler.add_noise(latents, torch.randn_like(latents), self.scheduler.timesteps[init_step])
        embeddings = torch.cat([self.embeddings['pos'].expand(batch_size, -1, -1), self.embeddings['neg'].expand(batch_size, -1, -1)])

        for i, t in enumerate(self.scheduler.timesteps[init_step:]):
    
            latent_model_input = torch.cat([latents] * 2)

            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=embeddings,
            ).sample

            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        imgs = self.decode_latents(latents) # [1, 3, 512, 512]
        return imgs

    def train_step(
        self,
        pred_rgb,
        step_ratio=None,
        guidance_scale=10,
        as_latent=False,
        vers=None, hors=None,
        name='default',
        save_error_name=None,
        mask=None,
        alpha_mask=None,
        res=256,
        nfsd=True
    ):
        
        batch_size = pred_rgb.shape[0]
        pred_rgb = pred_rgb.to(self.dtype)

        if as_latent:
            latents = F.interpolate(pred_rgb, (64, 64), mode="bilinear", align_corners=False) * 2 - 1
        else:
            # interp to 512x512 to be fed into vae.
            pred_rgb_512 = F.interpolate(pred_rgb, (res, res), mode="bilinear", align_corners=False)
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_512)

        with torch.no_grad():
            if step_ratio is not None:
                # dreamtime-like
                # t = self.max_step - (self.max_step - self.min_step) * np.sqrt(step_ratio)
                t = np.round((1 - step_ratio) * self.num_train_timesteps).clip(self.min_step, self.max_step)
                t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
                #print(t)
            else:
                t = torch.randint(self.min_step, self.max_step + 1, (batch_size,), dtype=torch.long, device=self.device)
            # w(t), sigma_t^2
            w = (1 - self.alphas[t]).view(batch_size, 1, 1, 1)

            # predict the noise residual with unet, NO grad!
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            tt = torch.cat([t] * 2)

            if hors is None:
                embeddings = torch.cat([self.embeddings['pos'].expand(batch_size, -1, -1), self.embeddings['neg'].expand(batch_size, -1, -1)])
            else:
                def _get_dir_ind(h):
                    if abs(h) < 60: return 'front'
                    elif abs(h) < 120: return 'side'
                    else: return 'back'
                    # if h == 0: return 'front'
                    # elif h == 180: return 'back'
                    # else: return 'side'
                    # if h == 0: return 'front'
                    # elif h == 180: return 'back'
                    # else: return 'side'

                embeddings = torch.cat([self.embeddings[_get_dir_ind(h)] for h in hors] + [self.embeddings['neg'].expand(batch_size, -1, -1)])
            #print(embeddings.shape, self.embeddings['pos'].expand(batch_size, -1, -1).shape,  self.embeddings['neg'].expand(batch_size, -1, -1).shape)
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                latent_model_input,
                tt,
                encoder_hidden_states=embeddings,
                controlnet_cond=torch.cat([self.control_image[name]] * 2),
                conditioning_scale=1.,
                guess_mode=False,
                return_dict=False,
            )

            #print(down_block_res_samples.shape, mid_block_res_sample.shape, latent_model_input.shape, embeddings.shape)
            noise_pred = self.unet(
                latent_model_input, tt, encoder_hidden_states=embeddings,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            ).sample

            # perform guidance (high scale from paper!)
            print(noise_pred.shape)
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )
            # noise_pred = guidance_scale * (
            #     noise_pred_cond - noise_pred_uncond
            # )

            # if t<200:    #delta_d
            #     #epsilon(zt,none,t) 
            #     noise_pred = noise_pred + noise_pred_uncond
            # else:
            #     #epsilon(zt,none,t)-epsilon(zt,neg,t) 
            #     noise_pred = noise_pred + (noise_pred_uncond-noise_pred_neg)
                

            grad = w * (noise_pred - noise)
            grad = torch.nan_to_num(grad)

            # seems important to avoid NaN...
            # grad = grad.clamp(-1, 1)

        target = (latents - grad).detach() # [1, 4, 64, 64]
        #print(target.shape, '???')

        # if mask is None:
        #     loss = 0.5 * F.mse_loss(latents.float(), target, reduction='sum') / latents.shape[0]
        # else:
        #     # mask weighted
        #     mask = F.interpolate(mask.unsqueeze(0), (target.shape[-2], target.shape[-1]), mode="nearest").detach()
        #     loss = 0.5 * torch.sum(F.mse_loss(latents.float(), target, reduction='none') * mask) / latents.shape[0]

        if alpha_mask is not None:
            # mask weighted
            #mask = F.max_pool2d(alpha_mask, alpha_mask.shape[-1]//res, alpha_mask.shape[-1]//res)
            #print(mask.shape, target.shape)
            mask = F.interpolate(alpha_mask, (target.shape[-2], target.shape[-1]), mode="bilinear").detach()
            loss = 0.5 * torch.sum(F.mse_loss(latents.float(), target, reduction='none') * mask) / latents.shape[0]
        elif mask is not None:
            # mask weighted
            # mask = F.interpolate(mask.unsqueeze(0), (target.shape[-2], target.shape[-1]), mode="nearest").detach()
            mask = F.interpolate(mask.unsqueeze(0), (target.shape[-2], target.shape[-1]), mode="bilinear").detach()
            loss = 0.5 * torch.sum(F.mse_loss(latents.float(), target, reduction='none') * mask) / latents.shape[0]
        else:
            loss = 0.5 * F.mse_loss(latents.float(), target, reduction='sum') / latents.shape[0]
       
        ###  debug error map ###
        if save_error_name is not None:
            error_map = (latents.float().detach() - target.float().detach()) ** 2
            error_map = torch.sum(error_map, dim=1).squeeze() # 64, 64
            error_map = error_map - torch.min(error_map)
            error_map = error_map / torch.max(error_map) # 0-1

            plt.imsave('/vision/u/xtiange/genhuman/gen-human/playground/' + save_error_name, error_map.detach().cpu().numpy(), cmap='jet')

        return loss

    @torch.no_grad()
    def produce_latents(
        self,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        latents=None,
    ):
        if latents is None:
            latents = torch.randn(
                (
                    1,
                    self.unet.in_channels,
                    height // 8,
                    width // 8,
                ),
                device=self.device,
            )

        batch_size = latents.shape[0]
        self.scheduler.set_timesteps(num_inference_steps)
        embeddings = torch.cat([self.embeddings['pos'].expand(batch_size, -1, -1), self.embeddings['neg'].expand(batch_size, -1, -1)])

        for i, t in enumerate(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=embeddings
            ).sample

            # perform guidance
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        return latents

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents

        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents

    def prompt_to_img(
        self,
        prompts,
        negative_prompts="",
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        latents=None,
    ):
        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        self.get_text_embeds(prompts, negative_prompts)
        
        # Text embeds -> img latents
        latents = self.produce_latents(
            height=height,
            width=width,
            latents=latents,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )  # [1, 4, 64, 64]

        # Img latents -> imgs
        imgs = self.decode_latents(latents)  # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype("uint8")

        return imgs


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str)
    parser.add_argument("--negative", default="", type=str)
    parser.add_argument(
        "--sd_version",
        type=str,
        default="2.1",
        choices=["1.5", "2.0", "2.1"],
        help="stable diffusion version",
    )
    parser.add_argument(
        "--hf_key",
        type=str,
        default=None,
        help="hugging face Stable diffusion model key",
    )
    parser.add_argument("--fp16", action="store_true", help="use float16 for training")
    parser.add_argument(
        "--vram_O", action="store_true", help="optimization for low VRAM usage"
    )
    parser.add_argument("-H", type=int, default=512)
    parser.add_argument("-W", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device("cuda")

    sd = StableDiffusion(device, opt.fp16, opt.vram_O, opt.sd_version, opt.hf_key)

    imgs = sd.prompt_to_img(opt.prompt, opt.negative, opt.H, opt.W, opt.steps)

    # visualize image
    plt.imshow(imgs[0])
    plt.show()
