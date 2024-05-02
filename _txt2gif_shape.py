import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_gif

device = torch.device("mps" if torch.cuda.is_available() else "cpu")

repo = "openai/shap-e"
pipe = DiffusionPipeline.from_pretrained(repo, torch_dtype=torch.float16)
pipe = pipe.to(device)

guidance_scale = 15.0
prompt = "a chair"

images = pipe(
    prompt,
    guidance_scale=guidance_scale,
    num_inference_steps=64,
    frame_size=256,
).images

gif_path = export_to_gif(images[0], "shark_3d.gif")