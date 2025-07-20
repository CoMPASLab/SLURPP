import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2
import torch
SCRATCH_DATA_DIR = os.environ.get('SCRATCH_DATA_DIR', "./")


class UnderwaterDataset(Dataset):
    def __init__(self, root_dir= f"{SCRATCH_DATA_DIR}", image_size=256, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_count = [None, None, None, None, None, None]  
        self.root_dir = root_dir
        self.image_size = image_size
        self.transform= transforms.Resize((image_size, image_size))
        self.image_paths = []
        count = 0
        for dirpath, dirs, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.png')):
                    self.image_paths.append(os.path.join(dirpath, file))
                    count += 1
        print(f"Found {count} images in {root_dir}")
        self.BL_dir = './datasets/BL_data_lab.npy'
        self.BL_data = np.load(self.BL_dir)
        self.BL_data = self.BL_data[self.BL_data[:, -1].argsort()]
        self.BL_data_subsets = [self.BL_data[self.BL_data[:, -1] == i] for i in range(10)]
        print(f"self.BL_data: {self.BL_data.shape}")
        print(f"self.BL_data_subsets: {len(self.BL_data_subsets)}")
        self.beta_c_table = [np.array([0.02206, 0.04805, 0.22922]),
                            np.array([0.02696, 0.05082, 0.23083]),
                            np.array([0.086, 0.1034, 0.2766]),
                            np.array([0.4818, 0.4339, 0.537]),
                            np.array([1.2935, 1.1107, 1.074]),
                            np.array([0.546, 0.463, 0.55]),
                            np.array([1.465, 1.228, 1.155]),
                            np.array([1.914, 1.567, 9.09]),
                            np.array([3.398, 2.773, 2.331]),
                            np.array([4.719, 3.81, 3.08])]
      

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            return self.getitem(idx)
        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            print(f"error image path: {self.image_paths[idx]}")
            u_image = torch.zeros((3, 256, 256), dtype=torch.float32)
            bc_image = torch.zeros((3, 256, 256), dtype=torch.float32)
            ill_image = torch.zeros((3, 256, 256), dtype=torch.float32)
            clear_image = torch.zeros((3, 256, 256), dtype=torch.float32)
            if self.transform:
                u_image = self.transform(u_image) # underwater image
                bc_image = self.transform(bc_image) # backscattered image
                ill_image = self.transform(ill_image) # illumiated image
                clear_image = self.transform(clear_image) # clear image

            # print(u_image.shape, u_image.dtype)
            out = {"imgs":{}}
            out['imgs']['u'] = u_image
            out['imgs']['bc'] = bc_image
            out['imgs']['ill'] = ill_image
            out['imgs']['clear'] = clear_image
            
            return out

    def getitem(self, idx):
        # print(f"getitem: {idx}")

        img_path = self.image_paths[idx]
        clear_image_srgb = Image.open(img_path).convert("RGB")
        img_arr = np.array(clear_image_srgb, dtype=np.float32) / 255.0

        clear_image = np.array(clear_image_srgb, dtype=np.float32)

        if img_path.lower().endswith('.png'):
            depth_map_path = img_path[:-4] + '_depth_pro.npy'
        elif img_path.lower().endswith('.jpg'):
            depth_map_path = img_path[:-4] + '_depth_pro.npy'
        try:
            depth_map = np.load(depth_map_path)
        except:
            print(f"Error loading depth map from {depth_map_path}, using default depth map.")
            depth_map = np.ones((img_arr.shape[0], img_arr.shape[1]), dtype=np.float32) * 10.0
        # print(depth_map.shape)

        beta_c = self.beta_c_table[np.random.randint(0, len(self.beta_c_table))]
        beta_c = np.array(beta_c)
        
        beta_c = beta_c[[2, 1, 0]]

        beta_c *= np.random.uniform(0.01, 1)

        if np.all(beta_c > 0.5):
            beta_c /= beta_c.max()
            beta_c *= np.random.uniform(0.1, 0.3)
        
        
        background_light_cands = self.BL_data_subsets[np.random.randint(0, len(self.BL_data_subsets))]
        background_light = background_light_cands[np.random.randint(0,background_light_cands.shape[0])][:3]
        

        ill_map = np.exp(-beta_c * np.expand_dims(depth_map, axis=2))

        backscatter_map = background_light * (1-ill_map)
        direct_light = clear_image * ill_map # low light
        u_img = direct_light + backscatter_map

        u_img_gray = cv2.cvtColor(u_img.astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0

        u_img_var = np.var(u_img_gray)
        if u_img_var < 0.0001:
            return self.__getitem__(np.random.randint(0, len(self)))
        
        u_img = u_img.astype(np.float32)/255.0
        u_img = u_img.clip(0, 1)

        backscatter_map = backscatter_map.astype(np.float32)/255.0

        clear_image = img_arr


        clear_image = torch.from_numpy(clear_image.clip(0, 1)).float().permute(2, 0, 1) # high light
        u_img = torch.from_numpy(u_img.clip(0, 1)).float().permute(2, 0, 1)
        ill_map = torch.from_numpy(ill_map.clip(0, 1)).float().permute(2, 0, 1)
        backscatter_map = torch.from_numpy(backscatter_map.clip(0, 1)).float().permute(2, 0, 1)
        if self.transform:
            clear_image = self.transform(clear_image)
            u_img = self.transform(u_img)
            ill_map = self.transform(ill_map)
            backscatter_map = self.transform(backscatter_map)
        

        # New code to randomly flip images
        if np.random.rand() < 0.5:  # 50% probability
            clear_image = transforms.functional.hflip(clear_image)
            u_img = transforms.functional.hflip(u_img)
            ill_map = transforms.functional.hflip(ill_map)
            backscatter_map = transforms.functional.hflip(backscatter_map)

        out = {"imgs":{}}
        out['imgs']['u'] = u_img
        out['imgs']['bc'] = backscatter_map
        out['imgs']['ill'] = ill_map
        out['imgs']['clear'] = clear_image
        

        return out
