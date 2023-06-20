import os,sys
import glob

import numpy as np
import pandas as pd

import torch.nn as nn 
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from collections import Counter

sys.path.append('.')
from XL.lib.utils.transforms import MinMaxScaler, AbsoluteScaler

class SwissImage(Dataset):
    '''Transformer needed to be added'''
    def __init__(self, dataset_csv, img_dir, dem_dir, mask_dir, common_transform=None, img_transform=None, debug=False):
        self.img_dir = img_dir
        self.dem_dir = dem_dir
        self.mask_dir = mask_dir
        self.img_dem_label = pd.read_csv(dataset_csv)
        if debug:
            self.img_dem_label = self.img_dem_label.iloc[:100]
        self.common_transform = common_transform
        self.img_transform = img_transform
        self.dem_max, self.dem_min = 4603, 948 
        self.dem_mean, self.dem_std = 0.4806, 0.2652
        self.mean = np.array([0.5585, 0.5771, 0.5543], dtype=np.float32)
        self.std = np.array([0.2535, 0.2388, 0.2318], dtype=np.float32)
        # Values for std and means (dem with absolute scaler)
        #self.mean = np.array([22339.53184346, 23085.64802353, 22170.41603125,            ], dtype=np.float32)
        #self.std = np.array([0.11846169, 0.11504443, 0.09361862,        ], dtype=np.float32)
        #self.dem_mean, self.dem_std = 0,1
        
    def __len__(self):
        return len(self.img_dem_label)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_dem_label.iloc[idx, 0])
        dem_path = os.path.join(self.dem_dir, self.img_dem_label.iloc[idx, 1])
        mask_path = os.path.join(self.mask_dir, self.img_dem_label.iloc[idx, 2])
        image = Image.open(img_path)
        
        ## resize rgb images from 400x400 to 200x200 to match the size of mask
        image = image.resize((200,200))
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
        
        dem_transform = transforms.Compose([
            transforms.ToTensor(),
            MinMaxScaler(self.dem_max, self.dem_min),
           # AbsoluteScaler(),
           transforms.Normalize(self.dem_mean, self.dem_std)
        ])
                                         
        image = basic_transform(image)
        dem = dem_transform(dem)
        mask = transforms.ToTensor()(mask)
        return image, dem, mask

    def _getImbalancedCount(self):
        count = Counter(self.img_dem_label['few'])
        return count
    
    def _getImbalancedClass(self, idx):
        return self.img_dem_label['few'][idx]

if __name__ =="__main__":
    
    dataset_csv =   '/home/valerie/Projects/Alps_LCC/data/split_subset/train_subset_few.csv'
        
    img_dir = '/home/valerie/data/rocky_tlm/rgb/'  #'/data/xiaolong/rgb'
    dem_dir = '/home/valerie/data/rocky_tlm/dem/' # /data/xiaolong/dem'
    mask_dir = '/home/valerie/data/ace_alps/mask'
    
    ds = SwissImage(dataset_csv,img_dir,dem_dir,mask_dir,common_transform=None,img_transform=None,debug=False)
    print(ds[0][0].shape,ds[0][1].shape,ds[0][2].shape,)
    

    