import os
import re
from datetime import datetime, timedelta
import imageio
import yaml
from config import Config
import svgwrite
from IPython.display import SVG, display, Image as IPythonImage

# Define utility functions

def ddyymm_hhmmss():
    return datetime.today().strftime('%y%m%d_%H%M%S')

def clean_text_prompt(prompt, max_length=70):
    '''Sanitizes a text prompt to be used as filenames.'''
    # Remove invalid filename characters
    clean_prompt = re.sub(r'\',[<>:"/\\|?*]', '', prompt)
    # Replace spaces with underscores
    clean_prompt = re.sub(r'\s+', '_', clean_prompt)
    # Optionally, truncate to avoid very long filenames
    clean_prompt = clean_prompt[:max_length]
    return clean_prompt

# Function to toggle colors based on dark_mode
def get_text_color():
    return Config.TXT_COLOR_DARK if Config.TXT_DARK_MODE else Config.TXT_COLOR_LIGHT

def get_background_color():
    return Config.TXT_COLOR_LIGHT if Config.TXT_DARK_MODE else Config.TXT_COLOR_DARK

def set_image_path():
    clean_prompt = clean_text_prompt(Config.PROMPT)
    Config.IMAGE_NAME = f'{ddyymm_hhmmss()}_{clean_prompt}_steps{Config.STEPS:03}.png'
    save_dir = os.path.join(Config.OUTPUT_DIR, Config.IMAGE_NAME)
    if Config.ALGO_TYPE and Config.ALGO_NAME:
      subdir = f'{Config.ALGO_TYPE}_{Config.ALGO_NAME}'.replace(' ', '-')
      save_dir = os.path.join(Config.OUTPUT_DIR, subdir)
    os.makedirs(save_dir, exist_ok=True)
    Config.IMAGE_PATH = os.path.join(save_dir, Config.IMAGE_NAME)

def save_meta(pipe):
    yml_path = f'{Config.IMAGE_PATH[:-4]}.yml'
    yml_pipe_path = f'{Config.IMAGE_PATH[:-4]}.pipe.yml'

    with open(yml_path, 'w') as file:
        yaml.dump(Config.to_dict(), file, default_flow_style=False)
    
    with open(yml_pipe_path, 'w') as file:
        yaml.dump(pipe.config, file, default_flow_style=False)

def save_image(image):
    image.save(Config.IMAGE_PATH)
    print(f'Saved image at {Config.IMAGE_PATH}.')
    return image

def save_gif(frames):
  save_path = f'{Config.IMAGE_PATH[:-4]}_{Config.FPS}fps.gif'
  imageio.mimsave(save_path, frames, fps=Config.FPS, loop=0)
  return save_path

def save_params_image(params, display_image=True):
  save_path = f'{Config.IMAGE_PATH[:-4]}.svg'

  top_margin = Config.TXT_FONT_SIZE

  # Get text
  text_left, text_right = get_svg_text(params)

  prompt_start_line = max(len(params.keys()), len(text_right)) + 1

  # Create drawing object
  dwg = svgwrite.Drawing(save_path, (Config.TXT_IMG_WIDTH, Config.TXT_IMG_HEIGHT), profile='tiny')

  # Set background color based on dark mode
  background_color = get_background_color()
  dwg.add(dwg.rect(insert=(0, 0), size=("100%", "100%"), fill=background_color))

  # Text properties
  font_family = Config.TXT_FONT
  font_size = str(Config.TXT_FONT_SIZE) + "px"
  text_color = get_text_color()

  # Draw left text
  y = Config.TXT_IMG_MARGIN + top_margin
  line_index = 0
  for line in text_left:
      text_color_override = text_color if prompt_start_line > line_index else Config.TXT_COLOR_MID
      dwg.add(dwg.text(line, insert=(Config.TXT_IMG_MARGIN, y), fill=text_color_override, font_family=font_family, font_size=font_size))
      y += Config.TXT_FONT_SIZE
      line_index += 1

  # Draw right text using text-anchor 'end' for right alignment
  y = Config.TXT_IMG_MARGIN + top_margin
  for line in text_right:
      x = Config.TXT_IMG_WIDTH - Config.TXT_IMG_MARGIN
      dwg.add(dwg.text(line, insert=(x, y), fill=text_color, font_family=font_family, font_size=font_size, text_anchor="end"))
      y += Config.TXT_FONT_SIZE + 5

  # Save SVG
  dwg.save()

  if display_image:
    display(SVG(save_path))

  return save_path

def ensure_output_dir():
    ensure_dir(Config.OUTPUT_DIR)

def ensure_dir(dir):
    '''Creates the given directories if they don't exist.'''
    if not os.path.exists(dir):
        os.makedirs(dir)

# Function to break long text into multiple lines with <tspan> elements
def break_text_for_svg(text, max_linelength):
    words = text.split(' ')
    lines = []
    current_line = ""

    for word in words:
        if len(current_line) + len(word) <= max_linelength:
            # Add the word to the current line
            current_line += f" {word}" if current_line else word
        else:
            # Add the current line to the lines list and start a new line
            lines.append(current_line)
            current_line = word

    # Add the last line to the lines list
    lines.append(current_line)

    # Create <tspan> elements for each line
    tspan_elements = [f"{line}" for line in lines]

    # Join the <tspan> elements to create the SVG text
    return "\n".join(tspan_elements)

def get_svg_text(params):
    # Get today's date
    timezone_offset = timedelta(hours=Config.TIME_ZONE)
    current_utc_datetime = datetime.utcnow()
    desired_datetime = current_utc_datetime + timezone_offset
    date = desired_datetime.strftime("%B %d, %Y")

    # Break long text into multiple lines with <tspan> elements
    broken_prompt = break_text_for_svg(Config.PROMPT, Config.TXT_MAX_LINELENGTH)

    text_right = [
        f'{Config.ALGO_NAME} ({Config.ALGO_TYPE})',
        Config.AUTHOR,
        date
    ]

    text_left = []

    # Add one parameter per line
    for key, value in params.items():
      text_left.append(f'{key} {value}')

    left_line_breaks = max(len(text_right) - len(params), 0) + 1
    for i in range(0, left_line_breaks):
      text_left.append('')

    # Add each line as a separate element in the list
    text_left.extend(broken_prompt.split('\n'))

    return text_left, text_right

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

def get_save_dir():
    return os.path.join(Config.OUTPUT_DIR, f'{Config.ALGO_TYPE}_{Config.ALGO_NAME}')