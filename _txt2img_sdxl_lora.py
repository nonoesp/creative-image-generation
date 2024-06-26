'''
Diffusion with a stable diffusion pipeline.
'''

# requires PEFT backend
# pip install peft

from diffusers import StableDiffusionXLPipeline
from utils import clean_text_prompt, ddyymm_hhmmss
import torch

# Config
STEPS = 35
SEED = 0
TORCH_DEVICE = 'mps'

# Create a Stable Diffusion pipeline
pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
pipe.to(TORCH_DEVICE)

# Load LoRA
pipe.load_lora_weights(
  "e-n-v-y/envy-anime-watercolor-xl-01",
  weight_name="EnvyAnimeWatercolorXL01.safetensors"
)

# Generate
prompt = 'anime style watercolor, a japanese house in fall'
generator = torch.Generator(TORCH_DEVICE).manual_seed(SEED)
image = pipe(prompt, num_inference_steps=STEPS, generator=generator).images[0]

# Save the image
cleaned_prompt = clean_text_prompt(prompt)
image.save(f'outputs/{ddyymm_hhmmss()}_{cleaned_prompt}_steps{STEPS:03}.png')