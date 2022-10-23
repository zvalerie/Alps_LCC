import os
import glob

import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torch.nn as nn 

def _resize(img, dem):
    c, h, w = img.size()
    dem = nn.functional.interpolate(dem.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False)
    return dem.squeeze(0)

class SwissImage(Dataset):
    '''Transformer needed to be added'''
    def __init__(self, dataset_csv, img_dir, dem_dir, mask_dir, transform=None, mask_transform=None, debug = False):
        self.img_dir = img_dir
        self.dem_dir = dem_dir
        self.mask_dir = mask_dir
        self.img_dem_label = pd.read_csv(dataset_csv)
        if debug:
            self.img_dem_label = self.img_dem_label.iloc[:100]
        self.transform = transform
        self.mask_transform = mask_transform
        self.mean = np.array([0.5405, 0.5583, 0.5364], dtype=np.float32)
        self.std = np.array([0.1254, 0.1201, 0.0961], dtype=np.float32)
        
    def __len__(self):
        return len(self.img_dem_label)
    
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_dem_label.iloc[idx, 0])
        dem_path = os.path.join(self.dem_dir, self.img_dem_label.iloc[idx, 1])
        mask_path = os.path.join(self.mask_dir, self.img_dem_label.iloc[idx, 2])
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        image = transform(Image.open(img_path))
        dem = transforms.ToTensor()(Image.open(dem_path))
        dem = _resize(image, dem) # upsampl dem [1, 200, 200] -> [1, 400, 400]
        mask = transforms.ToTensor()(Image.open(mask_path))
        
        if self.transform is not None:
            image = self.transform(image)
            dem = self.transform(dem)
        if self.mask_transform is not None:
            mask = self.transform(mask)
        return image, dem, mask
    