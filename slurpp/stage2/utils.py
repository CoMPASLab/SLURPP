# Authors: Mingyang Xie
from termcolor import colored
import os
import shutil
import time
import random
import os, random, time
import torch
import shutil
import numpy as np
import logging, warnings
import omegaconf
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
from termcolor import colored
from pynvml import *


def recursive_load_config(config_path: str) -> OmegaConf:
    conf = OmegaConf.load(config_path)

    output_conf = OmegaConf.create({})

    # Load base config. Later configs on the list will overwrite previous
    base_configs = conf.get("base_config", default_value=None)
    if base_configs is not None:
        assert isinstance(base_configs, omegaconf.listconfig.ListConfig)
        for _path in base_configs:
            assert (
                _path != config_path
            ), "Circulate merging, base_config should not include itself."
            _base_conf = recursive_load_config(_path)
            output_conf = OmegaConf.merge(output_conf, _base_conf)

    # Merge configs and overwrite values
    output_conf = OmegaConf.merge(output_conf, conf)

    return output_conf


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




def info(vector=None, name='', precision=4):
    """
    check info
    :param name: name
    :param vector: torch tensor or numpy array or list of tensor/np array
    """
    if torch.is_tensor(vector):
        if torch.is_complex(vector):
            print(colored(name, 'red') + f' tensor size: {vector.size()}, mean: {torch.mean(vector).item():.{precision}f}, dtype: {vector.dtype}, on Cuda: {vector.is_cuda}')
        else:
            try:
                print(colored(name, 'red') + f' tensor size: {vector.size()}, min: {torch.min(vector).item():.4f},  max: {torch.max(vector).item():.4f}, mean: {torch.mean(vector).item():.{precision}f}, dtype: {vector.dtype}, on Cuda: {vector.is_cuda}')
            except:
                print(colored(name, 'red') + f' tensor size: {vector.size()}, min: {torch.min(vector).item():.4f},  max: {torch.max(vector).item():.4f}, dtype: {vector.dtype}, on Cuda: {vector.is_cuda}')
    elif isinstance(vector, np.ndarray):
        try:
            print(colored(name, 'red') + f' numpy size: {vector.shape}, min: {np.min(vector):.4f},  max: {np.max(vector):.4f}, mean: {np.mean(vector):.4f}, dtype: {vector.dtype}')
        except:
            print(colored(name, 'red') + f' numpy size: {vector.shape}, min: {np.min(vector):.4f},  max: {np.max(vector):.4f}, dtype: {vector.dtype}')
    elif isinstance(vector, list):
        info(vector[0], f'{name} list of length: {len(vector)}, {name}[0]')
    else:
        print(colored(name, 'red') + 'Neither a torch tensor, nor a numpy array, nor a list of tensor/np array.' + f' type{type(vector)}')



def create_save_folder(folder_path, verbose=True):
    """
    1. check if there exists a folder with the same name,
    2. create folder
    3. save code in it
    :param folder_path: ex. /vulcanscratch/xxx/.../xxx/folder_name/
    :return: none
    """
    if 'debug' in folder_path.lower():
        if verbose:
            print("Removing '{:}'".format(folder_path))
        shutil.rmtree(os.path.abspath(folder_path), ignore_errors=True)
    if os.path.exists(folder_path) and 'debug' not in folder_path:
        if verbose:
            print("A folder with the same name already exists. \nPlease change a name for: " + colored(os.path.abspath(folder_path).split(os.sep)[-1], 'red'))
        # exit()

        print("If you wish to overwrite, please type: \"overwrite" + "\"")
        # print("If you wish to overwrite, please type: \"overwrite " + os.path.abspath(folder_path).split(os.sep)[-1] + "\"")
        timeout = time.time() + 10
        while True:
            val = input("Type here: ")
            if val != ("overwrite"):
                print("Does not match. Please type again or exit (ctrl+c).")
                # print("Chose not to overwrite. Exit the program.")
                # exit()
            else:
                print("Removing '{:}'".format(folder_path))
                shutil.rmtree(os.path.abspath(folder_path), ignore_errors=True)
                break
            if time.time() > timeout:
                print('timed out')
                exit()
    abs_save_path = os.path.abspath(folder_path)
    if verbose:
        print("Allocating '{:}'".format(colored(abs_save_path, 'red')))
    os.makedirs(abs_save_path)
    # os.makedirs(abs_save_path + '/logs')
    # print(abs_save_path)
    # exit()
    # copy_tree(os.getcwd(), abs_save_path + '/' + os.getcwd().split(os.sep)[-1])


def init_env():
    seed_torch(0)
    # nvmlInit()
    warnings.filterwarnings("ignore")
    os.environ["WANDB_SILENT"] = "True"
    logger = logging.getLogger("wandb")
    logger.setLevel(logging.ERROR)



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
