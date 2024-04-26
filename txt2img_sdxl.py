'''
Diffusion with a stable diffusion pipeline.
'''

from config import Config
from utils import save_meta, set_image_path, ensure_dir
from diffusers import StableDiffusionXLPipeline
import torch

# Parameters
Config.STEPS = 1
Config.PROMPT = 'a scenic landscape at dusk'

# Get image path
set_image_path()

# Create a Stable Diffusion pipeline
pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
pipe.to(Config.TORCH_DEVICE)

# Save metadata
save_meta(pipe)

# Generate
generator = torch.Generator(Config.TORCH_DEVICE).manual_seed(Config.SEED)
image = pipe(Config.PROMPT, num_inference_steps=Config.STEPS, generator=generator).images[0]

# Save the image
image.save(Config.IMAGE_PATH)