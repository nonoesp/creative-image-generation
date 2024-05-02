# https://huggingface.co/docs/diffusers/main/en/using-diffusers/lcm
import os
from diffusers import DiffusionPipeline, UNet2DConditionModel, LCMScheduler
import torch
from utils import clean_text_prompt, ddyymm_hhmmss, export_gif
from config import Config

# Settings
FPS = 10
GENERATION_STEPS = [15]
GENERATION_STEPS = range(1,16)
SEEDS = [100]
PROMPTS = ["hyper realistic photo of a polished alumninum building in the jungle, 8k, 16:9, masterpiece"]

# Load the Latent Consistency Models U-Net for fast generation.
unet = UNet2DConditionModel.from_pretrained(
    "latent-consistency/lcm-sdxl",
    torch_dtype=torch.float16,
    variant="fp16",
)

# Create a Stable Diffusion XL pipeline.
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    unet=unet,
    torch_dtype=torch.float16,
).to(Config.TORCH_DEVICE)

# Use the Latent Consistency Models scheduler for faster sampling.
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

for seed in SEEDS:
    for prompt in PROMPTS:
        frames = []
        generation_step = 1
        cleaned_prompt = clean_text_prompt(prompt)    
        for i in GENERATION_STEPS:
            inference_steps = i
            guidance_scale = 7.5
            generator = torch.Generator('mps').manual_seed(seed)
            ###
            image = pipe(
                prompt=prompt,
                num_inference_steps=inference_steps,
                generator=generator,
                guidance_scale=guidance_scale,
            ).images[0]
            if len(cleaned_prompt) > 30:
                cleaned_prompt = cleaned_prompt[:30]
            image_name = f'{ddyymm_hhmmss()}_lcm_seed{seed:04}_{cleaned_prompt}_steps{inference_steps:03}_guidance-scale{guidance_scale:03}_{generation_step:03}-of-{len(GENERATION_STEPS):03}.png'
            image.save(os.path.join(Config.OUTPUT_DIR, image_name))
            frames.append(image)
            generation_step += 1

        gif_name = f'{ddyymm_hhmmss()}_lcm-t2i_seed{seed:04}_{cleaned_prompt}_{len(GENERATION_STEPS):03}f.gif'
        gif_path = os.path.join(Config.OUTPUT_DIR, gif_name)
        export_gif(frames, gif_path, FPS)