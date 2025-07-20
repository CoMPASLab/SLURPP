from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
from termcolor import colored
import os, psutil, random, torch, warnings, logging, numpy as np, time, wandb
from pynvml import *

import torch
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T

def concat_images_with_labels(images, labels, font_path=None, font_size=20):
    # Check if the image list is not empty
    if len(images) == 0:
        raise ValueError("The images list is empty.")

    # Process images: clamp to [0, 1], convert to uint8, and transform to PIL images
    pil_images = []
    for img in images:
        # Clamp the tensor to range [0, 1]
        img = torch.clamp(img, 0, 1)
        # Convert the image to uint8 (range [0, 255])
        img = (img * 255).to(torch.uint8)
        # Convert torch tensor to PIL image
        pil_img = T.ToPILImage()(img)
        pil_images.append(pil_img)

    # Get the width and height of each image
    widths, heights = zip(*(img.size for img in pil_images))

    # Calculate total width and max height for concatenation
    total_width = sum(widths)
    max_height = max(heights)

    # Create a new image with a height for the text label area (font_size * 2 for padding)
    result_image = Image.new("RGB", (total_width, max_height + font_size * 2), (255, 255, 255))

    # Create a Draw object to add text later
    draw = ImageDraw.Draw(result_image)

    # Optional: Set a font if a font path is provided, otherwise use default
    if font_path:
        font = ImageFont.truetype(font_path, font_size)
    else:
        font = ImageFont.load_default()

    # Initialize x_offset for placing images side by side
    x_offset = 0

    for idx, pil_img in enumerate(pil_images):
        # Paste the current image into the result_image
        result_image.paste(pil_img, (x_offset, font_size * 2))

        # Add the corresponding label on top of each image
        label = labels[idx]
        bbox = draw.textbbox((0, 0), label, font=font)  # Get bounding box of the text
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        text_x = x_offset + (pil_img.width - text_width) // 2  # Center the label
        text_y = (font_size * 2 - text_height) // 2

        draw.text((text_x, text_y), label, font=font, fill=(0, 0, 0))

        # Update the x_offset for the next image
        x_offset += pil_img.width

    return result_image

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)


def init_env(use_wandb=True):
    seed_torch(0)
    warnings.filterwarnings("ignore")

    if use_wandb:
        logger = logging.getLogger("wandb")
        logger.setLevel(logging.ERROR)


def normalize(x):
    '''
    normalize the max value to 1, and min value to 0
    '''
    return (x - torch.min(x)) / (torch.max(x) - torch.min(x))