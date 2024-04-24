import imageio

# Config
INPUT_GIF_PATH = 'input.gif'
INPUT_GIF_PATH = '/Users/nono/repos/github.com/nonoesp/creative-image-generation/outputs/240424_002350_lcm-t2i_seed1011_a_hatefjall_ikea_office_beige_chair,_white_background,_4k_photo_059f.gif'
FPS = 10 # Set the desired fps here
OUTPUT_GIF_PATH = INPUT_GIF_PATH.replace('.gif', f'@{FPS}fps.gif')

# Read all frames from a GIF file
reader = imageio.get_reader(INPUT_GIF_PATH)

# Create a new GIF file with modified frames per second (fps)

with imageio.get_writer(OUTPUT_GIF_PATH, mode='I', fps=FPS) as writer:
    for frame in reader:
        writer.append_data(frame)

reader.close()