import os
import glob

import numpy as np
import pandas as pd

import torch.nn as nn 
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from collections import Counter

class SwissImage(Dataset):
    '''Transformer needed to be added'''
    def __init__(self, dataset_csv, img_dir, dem_dir, mask_dir, common_transform=None, img_transform=None, debug=False):
        self.img_dir = img_dir
        self.dem_dir = dem_dir
        self.mask_dir = mask_dir
        self.img_dem_label = pd.read_csv(dataset_csv)
        if debug:
            self.img_dem_label = self.img_dem_label.iloc[:1000]
        self.common_transform = common_transform
        self.img_transform = img_transform
        self.mean = np.array([0.5580, 0.5766, 0.5538], dtype=np.float32)
        self.std = np.array([0.1309, 0.1253, 0.1002], dtype=np.float32)
        
    def __len__(self):
        return len(self.img_dem_label)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_dem_label.iloc[idx, 0])
        dem_path = os.path.join(self.dem_dir, self.img_dem_label.iloc[idx, 1])
        mask_path = os.path.join(self.mask_dir, self.img_dem_label.iloc[idx, 2])
        image = Image.open(img_path)
        ## resize rgb images to match the size of mask
        image = image.resize((200,200),resample=Image.BILINEAR)
        dem = Image.open(dem_path)
        mask = Image.open(mask_path)
        
        if self.common_transform is not None:
            image, dem, mask = self.common_transform(image, dem, mask)

        if self.img_transform is not None:
            image = self.img_transform(image)
            
        basic_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        
        image = basic_transform(image)
        dem = transforms.ToTensor()(dem)
        mask = transforms.ToTensor()(mask)
        
        return image, dem, mask

    def _getImbalancedCount(self):
        count = Counter(self.img_dem_label['few'])
        return count
    
    def _getImbalancedClass(self, idx):
        return self.img_dem_label['few'][idx]