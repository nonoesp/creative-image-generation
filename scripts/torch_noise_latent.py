from PIL import Image
import torch

generator = torch.manual_seed(42)

tensor = torch.randn((3,64,64), generator=generator)

def save_tensor_as_image(tensor, filename):
    # Squeeze the batch dimension and select the first 3 channels if more are available
    tensor = tensor.squeeze(0)[:3, :, :]
    # Permute the tensor to put it into HxWxC format from CxHxW
    tensor = tensor.permute(1, 2, 0)
    # Clamp and normalize the tensor to a 0-255 range
    tensor = tensor.clamp(0, 1) * 255
    # Convert to numpy, ensure it's uint8 and then convert to a PIL Image
    image = Image.fromarray(tensor.cpu().numpy().astype('uint8'))
    # Save the image
    image.save(filename)

save_tensor_as_image(tensor, 'outputs/noise_latent.png')