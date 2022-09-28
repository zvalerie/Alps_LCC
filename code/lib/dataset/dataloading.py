import os
import glob

import numpy as np

from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import ToTensor

class SwissImage(Dataset):
    '''Transformer needed to be added'''
    def __init__(self, img_dir, dem_dir,img_label, transform=None, mask_transform=None):
        self.img_dir = img_dir
        self.dem_dir = dem_dir
        self.img_label = img_label 
        self.transform = transform
        self.mask_transform = mask_transform 
        # sorted() return a sorted list
        self.img_ids = sorted(img for img in os.listdir(self.img_dir))  
        self.img_dem = sorted(dem for dem in os.listdir(self.dem_dir)) 
        self.mask_ids = sorted(mask for mask in os.listdir(self.mask_ids))
        
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_ids[idx])
        dem_path = os.path.join(self.img_dir, self.img_ids[idx])
        mask_path = os.path.join(self.mask_ids, self.mask_ids[idx])
        image = ToTensor()(Image.open(img_path))
        dem = ToTensor()(Image.open(dem_path))
        mask = ToTensor()(Image.open(mask_path))
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.transform(mask)
        return image, dem, mask