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

# download an image
image = load_image(
    "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
)
image = np.array(image)

# Get canny image
image = cv2.Canny(image, 100, 200)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

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

# Generate
generator = torch.manual_seed(0)
image = pipe(
    "futuristic-looking woman with a blue sky background at dawn",
    num_inference_steps=20,
    generator=generator,
    image=canny_image,
).images[0]

# Save
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
image_path = os.path.join(OUTPUT_DIR, f'{ddyymm_hhmmss()}_controlnet.png')
image.save(image_path)
canny_image.save(image_path.replace('.png', '_canny.png'))