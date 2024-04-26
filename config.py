from utils import get_torch_device

class Config:
    ### Global config
    AUTHOR = 'Nono Martínez Alonso'
    TORCH_DEVICE = get_torch_device(fallback='cpu')
    OUTPUT_DIR = 'outputs'
    
    ### Default parameters
    SEED = 0
    STEPS = 5
    PROMPT = "a scenic landscape"
    
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
        }    