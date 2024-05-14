from utils import ddyymm_hhmmss, clean_text_prompt
from config import Config
import os

# Config
WEIGHTS_DIR = 'ostris/ikea-instructions-lora-sdxl'
WEIGHTS_NAME = 'ikea_instructions_xl_v1_5.safetensors'
TORCH_DEVICE = 'mps'

from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16).to(TORCH_DEVICE)

# Load LoRA weights
pipeline.load_lora_weights(WEIGHTS_DIR, weight_name=WEIGHTS_NAME)

# Generate an image
PROMPT = "a jeff koons balloon sculpture ikea instructions"
SEED = 4
STEPS = 50
SEEDS = range(100,112)
STEPS = 50
clean_prompt = clean_text_prompt(PROMPT)

for seed in SEEDS:
    generator = torch.Generator(TORCH_DEVICE).manual_seed(seed)
    image = pipeline(PROMPT, generator=generator, num_inference_steps=STEPS).images[0]
    image_name = f'{ddyymm_hhmmss()}_lora-{WEIGHTS_NAME}_{clean_prompt}_scale1.0_seed{seed:04}_steps{STEPS}.png'
    image.save(os.path.join(Config.OUTPUT_DIR, image_name))