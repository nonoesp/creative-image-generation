'''
Classifier-free guidance (CFG) scale.
'''

from diffusers import StableDiffusionPipeline
from utils import clean_text_prompt, ddyymm_hhmmss
import torch
import os
import imageio

from utils import remap
y = remap([x*x/5 for x in range(0,30)], [1.0, 8.0])
print(list(y))

# Config
STEPS = 25
SEEDS = [100234]#[123,100,1000,456,3049493]
TORCH_DEVICE = 'mps'
GUIDANCE_SCALES = y#[0,0.2,0.3,0.5,0.7,1,2,3,4,5,6,7,8]
PROMPTS = ["An isolated concrete building in Tokyo, Japan, in a scene landscape with nothing arround it, seen from the distance, high quality image"]
OUTPUT_DIR = 'outputs'
FPS = 20

# Create a Stable Diffusion pipeline
print('› Loading pipeline...')
pipe = StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5')
pipe.to(TORCH_DEVICE)
print('› Loaded...')

# Generate
for seed in SEEDS:
    for prompt in PROMPTS:
        frames = []
        cleaned_prompt = clean_text_prompt(prompt)
        for guidance_scale in GUIDANCE_SCALES:
            generator = torch.Generator(TORCH_DEVICE).manual_seed(seed)
            image = pipe(
                prompt,
                num_inference_steps=STEPS,
                generator=generator,
                guidance_scale=guidance_scale,
            ).images[0]

            # Save the image
            image_name = f'{ddyymm_hhmmss()}_S{seed:03}_{cleaned_prompt}_steps{STEPS:03}_cfg{guidance_scale}.png'
            image.save(os.path.join(OUTPUT_DIR, image_name))
            frames.append(image)
        
        if len(frames) > 1:
            gif_name = f'{ddyymm_hhmmss()}_cfg_seed{seed:04}_{cleaned_prompt}_cfg{GUIDANCE_SCALES[:1]}-{GUIDANCE_SCALES[-1:]}f.gif'
            imageio.mimsave(os.path.join(OUTPUT_DIR, gif_name), frames, fps=FPS, loop=0)