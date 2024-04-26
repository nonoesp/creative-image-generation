import os
import re
from datetime import datetime
import imageio
import yaml
from config import Config

def set_image_path():
    ensure_output_dir()
    clean_prompt = clean_text_prompt(Config.PROMPT)
    Config.IMAGE_NAME = f'{ddyymm_hhmmss()}_{clean_prompt}_steps{Config.STEPS:03}.png'
    Config.IMAGE_PATH = os.path.join(Config.OUTPUT_DIR, Config.IMAGE_NAME)

def ensure_output_dir():
    ensure_dir(Config.OUTPUT_DIR)

def ddyymm_hhmmss():
    return datetime.today().strftime('%y%m%d_%H%M%S')

def clean_text_prompt(prompt, max_length=30):
    '''Sanitizes a text prompt to be used as filenames.'''
    # Remove invalid filename characters
    clean_prompt = re.sub(r'\',[<>:"/\\|?*]', '', prompt)
    # Replace spaces with underscores
    clean_prompt = re.sub(r'\s+', '_', clean_prompt)
    # Optionally, truncate to avoid very long filenames
    clean_prompt = clean_prompt[:max_length]
    return clean_prompt

def ensure_dir(dir):
    '''Creates the given directories if they don't exist.'''
    if not os.path.exists(dir):
        os.makedirs(dir)

def export_gif(frames, fps, save_path):
    if len(frames) > 1:
        imageio.mimsave(save_path, frames, fps=fps, loop=0)

def export_montage():
    print('export_montage: Not implemented')

def remap_number(n, from_range, to_range):
    '''Remaps each number from the `from_range` to the `to_range`.'''
    from_min, from_max = from_range
    to_min, to_max = to_range
    return ((n - from_min) / (from_max - from_min)) * (to_max - to_min) + to_min

def remap(list, to_range):
    '''Remaps a given number list to a given `to_range`, e.g., `[100, 200]`.'''
    # Determine the minimum and maximum values in the list
    min_val = min(list)
    max_val = max(list)

    # Apply the function to each number in the list
    return [remap_number(n, [min_val, max_val], to_range) for n in list]

def save_meta(pipe):
    yml_path = f'{Config.IMAGE_PATH}.yml'
    yml_pipe_path = f'{Config.IMAGE_PATH}.pipe.yml'

    with open(yml_path, 'w') as file:
        yaml.dump(Config.to_dict(), file, default_flow_style=False)
    
    with open(yml_pipe_path, 'w') as file:
        yaml.dump(pipe.config, file, default_flow_style=False)