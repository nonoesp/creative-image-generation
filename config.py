def get_config_property(prop_name):
    return getattr(Config, prop_name, None)

class Config:
    ### Global config
    AUTHOR = None
    TORCH_DEVICE = 'cuda' # 'cpu' if no GPU is available, 'mps' for Apple silicon
    OUTPUT_DIR = '/content/drive/MyDrive/IAAC/GenAI/Outputs'
    TIME_ZONE = -5 # Relative to UTC time
    ALGO_TYPE = None
    ALGO_NAME = None

    ### Default parameters
    SEED = 0
    STEPS = 5
    PROMPT = "a scenic landscape"

    ### Automatic variables
    IMAGE_NAME = None
    IMAGE_PATH = None

    ### GIF Settings
    FPS = 6

    ### Params image settings
    TXT_IMG_WIDTH = 1024
    TXT_IMG_HEIGHT = 768
    TXT_IMG_MARGIN = 0
    TXT_DARK_MODE = False
    TXT_COLOR_LIGHT = 'black'
    TXT_COLOR_DARK = 'white'
    TXT_COLOR_MID = 'gray'
    TXT_FONT_SIZE = 44
    TXT_MAX_LINELENGTH = 48
    TXT_FONT = "Fira Sans" # "Courier New"

    def check():
      settings_to_check = [
         'PROMPT'
         'AUTHOR',
         'ALGO_TYPE',
         'ALGO_NAME',
      ]
      for prop_name in settings_to_check:
        prop = get_config_property(prop_name)
        if prop is None:
          raise Exception(f'Config.{prop_name} needs to be set.')
      print('Config OK.')

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