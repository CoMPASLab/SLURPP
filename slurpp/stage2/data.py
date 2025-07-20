# Authors: Mingyang Xie, Tianfu Wang

from torch.utils.data import DataLoader, Dataset
import os
from utils import *
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

class ImageDataset_Decoder(Dataset):
    def __init__(self, data_folder, field, resize = True, img_size_out = 384):
        super(ImageDataset_Decoder, self).__init__()
        self.data_folder = data_folder
        self.img_dirs = [entry for entry in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, entry))]
        self.size = len(self.img_dirs)
        self.data_folder = data_folder
        self.field = field

        self.to_tensor = transforms.ToTensor()
        
        self.img_size_out = img_size_out
        if resize:
            self.resize_img = True
            self.resize = transforms.Resize((img_size_out, img_size_out))
        else:
            self.resize_img = False



    def __getitem__(self, index):

        output = {}
        img_dir = self.img_dirs[index]

        try:
            target_img = Image.open(f"{self.data_folder}/{img_dir}/{self.field}/output_gc.png")
            composite_img = Image.open(f"{self.data_folder}/{img_dir}/composite_img.png")
            target_img = self.to_tensor(target_img)
            composite_img = self.to_tensor(composite_img)

            blurry_image = Image.open(f"{self.data_folder}/{img_dir}/{self.field}/output_pred_gc.png")
            blurry_image = self.to_tensor(blurry_image)
              
                
            output['target_img'] =  target_img
            output['composite_img'] =  composite_img
            output['blurry_img'] = blurry_image

            if self.resize_img:
                output['target_img'] = self.resize(output['target_img'])
                output['composite_img'] = self.resize(output['composite_img'])
                output['blurry_img'] = self.resize(output['blurry_img'])  

        except Exception as e:
            print(f"Warning: Could not load image in directory {img_dir}. Error: {e}")

            output['target_img'] = torch.zeros(3, self.img_size_out, self.img_size_out)  # Placeholder image
            output['composite_img'] = torch.zeros(3, self.img_size_out, self.img_size_out)  # Placeholder image
            output['blurry_img'] = torch.zeros(3, self.img_size_out, self.img_size_out)  # Placeholder image


        return output
    
    def __len__(self):
        return self.size


