from diffusers import AutoPipelineForText2Image
import torch

pipe = AutoPipelineForText2Image.from_pretrained("kandinsky-community/kandinsky-3", variant="fp16", torch_dtype=torch.float16)
pipe.enable_model_cpu_offload()

prompt = "A photograph of the inside of a subway train. There are animals sitting on the seats. One of them is reading a newspaper. The window shows the city in the background."

generator = torch.Generator(device="mps").manual_seed(0)
image = pipe(prompt, num_inference_steps=25, generator=generator).images[0]