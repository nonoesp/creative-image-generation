import torch
import sys

class Config:
    ### Global config
    AUTHOR = 'Nono Martínez Alonso'
    TORCH_DEVICE = 'cuda' if 'google.colab' in sys.modules else 'mps' if torch.backends.mps.is_available() else 'cpu'
    OUTPUT_DIR = 'outputs'
    
    ### Default parameters
    SEED = 0
    STEPS = 5
    PROMPT = "a scenic landscape"
    
    ### Settings
    FPS = 20

    ### Automatic variables
    IMAGE_NAME = 'image.png'
    IMAGE_PATH = 'image.png'
    
    def to_dict():
        return {
            'output_dir': Config.OUTPUT_DIR,
            'torch_device': Config.TORCH_DEVICE,
            'author': Config.AUTHOR,
            'image_name': Config.IMAGE_NAME,
            'image_path': Config.IMAGE_PATH,
            'prompt': Config.PROMPT,
            'seed': Config.SEED,
            'fps': Config.FPS
        }