# https://huggingface.co/docs/diffusers/en/api/pipelines/controlnet
# !pip install opencv-python transformers accelerate
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import numpy as np
import torch

import cv2
from PIL import Image
import os

from utils import clean_text_prompt, ddyymm_hhmmss

# Config
TORCH_DEVICE = 'mps'
OUTPUT_DIR = 'outputs'
# SEEDS = range(100,110)
SEEDS = [107]
STEPS = 60
# PROMPT = "a lighthouse by the sea, raw image, uhd, mirrorless photo"
# canny_image = load_image("outputs/240423_020904_manual-canny.png")

PROMPT = "an eastern island stone statue at the British Museum surrounded by tourists, 8k, masterpiece"
PROMPT = "an eastern island stone statue at the British Museum surrounded by people visiting the museum taking photos, 8k, masterpiece"
canny_image = load_image("inputs/sketch-190419_london-british-museum-hoa-hokakanaia_canny.png")
# download an image
# image = np.array(canny_image)

# Get canny image
# image = cv2.Canny(image, 100, 200)
# image = image[:, :, None]
# image = np.concatenate([image, image, image], axis=2)
# canny_image = Image.fromarray(image)

# Load control net and stable diffusion v1-5
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16,
).to(TORCH_DEVICE)

# Speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# Remove following line if xformers is not installed or not using on cuda device.
# pipe.enable_xformers_memory_efficient_attention()
# pipe.enable_model_cpu_offload()

cleaned_prompt = clean_text_prompt(PROMPT)
canny_image.save(f'outputs/{ddyymm_hhmmss()}_controlnet_input.png')

# Generate
for seed in SEEDS:
    generator = torch.manual_seed(seed)
    image = pipe(
        PROMPT,
        num_inference_steps=STEPS,
        generator=generator,
        image=canny_image,
    ).images[0]

    # Save
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    image_name = f'{ddyymm_hhmmss()}_controlnet_seed{seed:04}_{cleaned_prompt}_steps{STEPS:03}.png'
    image_path = os.path.join(OUTPUT_DIR, image_name)
    image.save(image_path)