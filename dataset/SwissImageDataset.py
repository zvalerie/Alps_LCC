import os,sys
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn 
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from collections import Counter

sys.path.append('.')
from utils.transforms import MinMaxScaler, AbsoluteScaler

class SwissImage(Dataset):
    '''Transformer needed to be added'''
    def __init__(self, dataset_csv, img_dir, dem_dir, label_dir, common_transform=None, img_transform=None, debug=False):
        self.img_dir = img_dir
        self.dem_dir = dem_dir
        self.label_dir = label_dir
        self.img_dem_label = pd.read_csv(dataset_csv)
        if debug:
            self.img_dem_label = self.img_dem_label.iloc[:250]
        self.common_transform = common_transform
        self.img_transform = img_transform
        self.dem_max, self.dem_min = 4603, 948 
        self.dem_mean, self.dem_std = 0.4806, 0.2652
        self.mean = np.array([0.5585, 0.5771, 0.5543], dtype=np.float32)
        self.std = np.array([0.2535, 0.2388, 0.2318], dtype=np.float32)

        
    def __len__(self):
        return len(self.img_dem_label)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_dem_label.iloc[idx, 0])
        dem_path = os.path.join(self.dem_dir, self.img_dem_label.iloc[idx, 1])
        label_path = os.path.join(self.label_dir, self.img_dem_label.iloc[idx, 2])
        image = Image.open(img_path)
        
        ## resize rgb images from 400x400 to 200x200 to match the size of label
        image = image.resize((200,200))
        dem = Image.open(dem_path)
        label = Image.open(label_path)
        
        if self.common_transform is not None:
            image, dem, label = self.common_transform(image, dem, label)

        if self.img_transform is not None:
            image = self.img_transform(image)
            
        basic_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        
        dem_transform = transforms.Compose([
            transforms.ToTensor(),
          #  AbsoluteScaler(),
            MinMaxScaler(self.dem_max, self.dem_min),
           transforms.Normalize(self.dem_mean, self.dem_std)
        ])
                                         
        image = basic_transform(image)
        dem = dem_transform(dem)
        label = transforms.ToTensor()(label)
        return image, dem, label

    def _getImbalancedCount(self):
        count = Counter(self.img_dem_label['few'])
        return count
    
    def _getImbalancedClass(self, idx):
        return self.img_dem_label['few'][idx]
    
    


def xunnormalize_batch(batch):
    """
    Unnormalizes a batch of tensors.

    Args:
        batch (torch.Tensor): Batch of tensors to be unnormalized, of shape (B, C, H, W).
        mean (sequence): Sequence of mean values for each channel.
        std (sequence): Sequence of standard deviation values for each channel.

    Returns:
        torch.Tensor: Unnormalized batch of tensors.
    """
    mean = [0.5585, 0.5771, 0.5543]  
    std = [0.2535, 0.2388, 0.2318] 
    # Create a normalization tensor of the same shape as the batch
    norm_tensor = torch.Tensor(mean)[None, :, None, None]  # Shape: (1, C, 1, 1)
    std_tensor = torch.Tensor(std)[None, :, None, None]    # Shape: (1, C, 1, 1)

    # Unnormalize the batch
    unnormalized_batch = batch * std_tensor + norm_tensor

    return unnormalized_batch

def unnormalize_batch(images, ):
    # Assuming images is a NumPy array with shape (batch_size, height, width, channels)
    # mean and std should be lists or arrays with length equal to the number of channels
    mean = [0.5585, 0.5771, 0.5543]  
    std = [0.2535, 0.2388, 0.2318] 

    unnormalized_images = torch.empty_like(images)
    if len(images.shape ) == 4: 
        for i in range(len(mean)):
            unnormalized_images[:, i, :, ] = (unnormalized_images[:, i, :, :] * std[i]) + mean[i]
    elif len(images.shape) == 3 :
         for i in range(len(mean)):
                unnormalized_images[ i, :, ] = (unnormalized_images[ i, :, :] * std[i]) + mean[i]
          
    return unnormalized_images   



if __name__ =="__main__":
    
    dataset_csv =   '/home/valerie/Projects/Alps_LCC/data/split_subset/train_subset_few.csv'
        
    img_dir = '/home/valerie/data/rocky_tlm/rgb/'  #'/data/xiaolong/rgb'
    dem_dir = '/home/valerie/data/rocky_tlm/dem/' # /data/xiaolong/dem'
    label_dir = '/home/valerie/data/ace_alps/label'
    
    ds = SwissImage(dataset_csv,img_dir,dem_dir,label_dir,common_transform=None,img_transform=None,debug=False)
    print(ds[0][0].shape,ds[0][1].shape,ds[0][2].shape,)
    

    