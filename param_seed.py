'''
Same seed. Different prompt.
'''

from diffusers import StableDiffusionPipeline
from utils import clean_prompt, ddyymm_hhmmss
import torch

# Config
STEPS = 15
SEED = 0
TORCH_DEVICE = 'mps'
# PROMPT = "a standalone isolated minimalist building in the SEASON in tokyo, japan, raw photo, uhd"
# VARIANTS = ["fall", "winter", "spring", "summer"]
PROMPT = "a box of FRUITs, central perspective, oil-painting, uhd"
VARIANTS = {
    'FRUIT': ["banana", "orange", "apple", "pear", "peach"],
}

# Create a Stable Diffusion pipeline
print('› Loading pipeline...')
pipe = StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5')
pipe.to(TORCH_DEVICE)
print('› Loaded...')

# Prompt variations
prompts = []
for key in VARIANTS.keys():
    for variant in VARIANTS[key]:
        prompts.append(PROMPT.replace(key, variant))

print(prompts)

prompts = [
    "a dark glass of gazpacho",
    "a wine glass of gazpacho",
    "a self portrait oil painting of a fireman",
    "a self portrait oil painting of a fireman, dark",
    "a self portrait oil painting of a fireman, light",
    "a japanese house standalone",
    "a japanese house standalone, concrete",
    "a japanese house standalone, wood",
    "a cat",
    "a cat and a dog",
    "a motorcycle, red",
    "a motorcycle, blue",
]

# Generate
for prompt in prompts:
    generator = torch.Generator(TORCH_DEVICE).manual_seed(SEED)
    image = pipe(
        prompt,
        num_inference_steps=STEPS,
        generator=generator,
        guidance_scale=1.0
    ).images[0]

    # Save the image
    cleaned_prompt = clean_prompt(prompt)
    image.save(f'outputs/{ddyymm_hhmmss()}_{cleaned_prompt}_steps{STEPS:03}.png')