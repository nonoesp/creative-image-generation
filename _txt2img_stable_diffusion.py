'''
Diffusion with a stable diffusion pipeline.
'''

from diffusers import StableDiffusionPipeline
from utils import clean_text_prompt, ddyymm_hhmmss
import torch

# Config
STEPS = 35
SEED = 0
TORCH_DEVICE = 'mps'

# Create a Stable Diffusion pipeline
pipe = StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5')
pipe.to(TORCH_DEVICE)

# Generate
generator = torch.Generator(TORCH_DEVICE).manual_seed(SEED)
prompt = "a prairie with mountains in the background"
image = pipe(prompt, num_inference_steps=STEPS, generator=generator).images[0]

# Save the image
cleaned_prompt = clean_text_prompt(prompt)
image.save(f'outputs/{ddyymm_hhmmss()}_{cleaned_prompt}_steps{STEPS:03}.png')