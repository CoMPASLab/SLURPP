import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np


class UnderwaterRealDataset(Dataset):
    def __init__(self, root_dir: str, image_size=256):
        self.root_dir = root_dir
        self.transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])

        self.image_files = []
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if os.path.isfile(os.path.join(subdir, file)):
                    self.image_files.append(os.path.join(subdir, file))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        try:
            return self.getitem(idx)
        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            u_image = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))  # Placeholder for underwater image
            bc_image = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))  # Placeholder for backscattered image
            ill_image = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))  # Placeholder for illuminated image
            clear_image = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))  # Placeholder for clear image
            if self.transform:
                u_image = self.transform(u_image)  # underwater image
                bc_image = self.transform(bc_image)  # backscattered image
                ill_image = self.transform(ill_image)  # illumiated image
                clear_image = self.transform(clear_image)  # clear image

            print(u_image.shape, u_image.dtype)
            out = {"imgs": {}}
            out['imgs']['u'] = u_image
            out['imgs']['bc'] = bc_image
            out['imgs']['ill'] = ill_image
            out['imgs']['clear'] = clear_image

            return out

    def getitem(self, idx):
        u_img_name = self.image_files[idx]
        u_image = Image.open(u_img_name)

        # Store original dimensions before transformation
        original_width, original_height = u_image.size

        if self.transform:
            u_image = self.transform(u_image)  # underwater image

        out = {"imgs": {}}

        out['imgs']['u'] = u_image
        out['imgs']['name'] = os.path.splitext(os.path.basename(u_img_name))[0]
        out['imgs']['original_size'] = (original_width, original_height)

        return out
