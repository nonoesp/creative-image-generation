# Config
WEIGHTS_DIR = 'weights/'
WEIGHTS_NAME = 'pytorch_lora_weights_05000.safetensors'
TORCH_DEVICE = 'cpu'

from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to(TORCH_DEVICE)
pipeline.load_lora_weights(WEIGHTS_DIR, weight_name=WEIGHTS_NAME)

# Use nnmsktch in the prompt to trigger the LoRA
PROMPT = "nnmsktch ink drawing of an elephant, white background"
SEED = 4
generator = torch.Generator(TORCH_DEVICE).manual_seed(SEED)
image = pipeline(PROMPT, generator=generator, num_inference_steps=50).images[0]
image