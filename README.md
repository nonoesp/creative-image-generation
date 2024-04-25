# Creative Image Generation

Creative image generation.

## Creating a Python environment

Most samples in this repository should work in Google Colab free, but it's convenient to set up a local environment to avoid installing Python libraries and downloading models on every session.

Here's how to create a Python environment with Anaconda.

```sh
# Create an Anaconda environment
conda create -n diffusers -y python=3.11

# Activate the environment
conda activate diffusers

# Install dependencies
pip install diffusers transformers torch accelerate

# Optionally, install useful libraries
pip install scipy imageio matplotlib opencv-python
```

## General utils

- timestamps
- gif making
- montages
- etc
- iteration through - latent space tensors, diffusion process, final diffusion steps, parameter interpolation (guidance scale, image control)

## HuggingFace Diffusers

The [Diffusers](https://huggingface.co/docs/diffusers/index) library developed by HuggingFace makes it easy to interact with state-of-the-art diffusion models to generate images, audio, and 3D objects.
It wraps existing models in so-called pipelines for inference and training.

## Diffusion concepts

- guidance
- conditioning / unconditioning
- noise latents
- denoising
- fine-tuning
- training
- scheduling /sampling / convergence

## Diffusion parameters

There are a series of Python scripts with the `param` prefix on their name that serve as a an example to understand each of the following concepts.

- steps
- [seed](param_seed.py)
- [guidance_scale](param_guidance_scale.py)
- prompt
- 

## Latent Consistency Models for Fast Generation

TK.

## Image-based guidance with ControlNet

TK.

- Depth
- Edges

## Training vs. fine-tuning

TK.

## Fine-tuning Stable Diffusion with LoRA (Low-Raw Adaptation)

TK.

### Dataset preparation

TK.