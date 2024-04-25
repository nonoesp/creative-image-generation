'''
Same seed. Different prompt.
'''

from diffusers import StableDiffusionPipeline
from utils import clean_text_prompt, ddyymm_hhmmss
import torch

# Config
STEPS = 25
SEED = 0
TORCH_DEVICE = 'mps' # Set this to 'cuda' if in Colab with GPU, otherwise 'cpu'

# Define a set of prompt and variants.
PROMPT = "a standalone isolated minimalist building in the SEASON in tokyo, japan, scenic landscape, at dawn, 4k, hd, raw photo, uhd"
VARIANTS = {
    'SEASON': ["fall", "winter", "spring", "summer"],
}

# Another example of possible prompt and variants.
# PROMPT = "a box of FRUITs, central perspective, oil-painting, uhd"
# VARIANTS = {
#     'FRUIT': ["banana", "orange", "apple", "pear", "peach"],
# }

# Create the prompt variations.
prompts = []
for key in VARIANTS.keys():
    for variant in VARIANTS[key]:
        prompts.append(PROMPT.replace(key, variant))

# Create a Stable Diffusion pipeline.
pipe = StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5')
pipe.to(TORCH_DEVICE)

# Generate
for prompt in prompts:
    print(prompt)
    generator = torch.Generator(TORCH_DEVICE).manual_seed(SEED)
    image = pipe(
        prompt,
        num_inference_steps=STEPS,
        generator=generator
    ).images[0]

    # Save the image
    cleaned_prompt = clean_text_prompt(prompt)
    image.save(f'outputs/{ddyymm_hhmmss()}_{cleaned_prompt}_steps{STEPS:03}.png')