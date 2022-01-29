import os
from glob import glob
from torch.utils.data import Dataset
from PIL import Image

from utils.constant import ATTRIBUTES

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.list_img = glob(os.path.join(img_dir, '*.png')) + glob(os.path.join(img_dir, '*.jpg'))
        self.list_img.sort()
        self.transform = transform
        self.attr_dict = ATTRIBUTES
    
    def __len__(self):
        return len(self.list_img)

    def __getitem__(self, idx):
        img_path = self.list_img[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, os.path.basename(img_path)
