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
- saving metadata

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

### Dataset preparation

- [ ] How to create an image captioning dataset

## Fine-tuning Stable Diffusion with LoRA (Low-Raw Adaptation)

- [ ] How to train stable diffusion LoRAs

### TODO

- [ ] Scripts to explain each parameter
    - [ ] param_width_height.py
    - [ ] ...
- [ ] Method to store generation parameters in YAML or JSON
- [ ] A utility library
  - [ ] animated gifs
  - [ ] montages
  - [ ] cleaning prompt names
  - [ ] file names

### Python tricks

Generating series of numbers.

```sh
» [x/10 for x in range(0,11)]
# [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

» [x for x in range(0,110,10)]
# [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100] 

» 
```