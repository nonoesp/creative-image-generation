import PIL
import requests
import torch
from io import BytesIO

from diffusers import StableDiffusionInstructPix2PixPipeline
from utils import clean_text_prompt, ddyymm_hhmmss
import os

# Config
STEPS = 15
SEEDS = [0,123,100,1000,456,3049493]
TORCH_DEVICE = 'mps'
GUIDANCE_SCALES = [0.3,0.5,0.7,1,3,5,7,8]
PROMPTS = ["make the mountains snowy"]
OUTPUT_DIR = 'outputs'

def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")


img_url = "https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png"

image = download_image(img_url).resize((512, 512))

pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    "timbrooks/instruct-pix2pix", torch_dtype=torch.float16
)
pipe = pipe.to(TORCH_DEVICE)

for prompt in PROMPTS:
    cleaned_prompt = clean_text_prompt(prompt)    
    for seed in SEEDS:
        for guidance_scale in GUIDANCE_SCALES:
            image = pipe(prompt=prompt, image=image).images[0]

            image_name = f'{ddyymm_hhmmss()}_instructpix2pix_S{seed:03}_{cleaned_prompt}_steps{STEPS:03}_cfg{guidance_scale}.png'
            image.save(os.path.join(OUTPUT_DIR), image_name)