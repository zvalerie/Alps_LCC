
import os
from tqdm import tqdm
from random import shuffle
import torch
from SwissImageDataset import SwissImage
import numpy as np
from torchvision import transforms
from utils.transforms import Compose, MyRandomRotation90, MyRandomHorizontalFlip, MyRandomVerticalFlip


def compute_mean_std(data_dir=None, full=True):
    
    print('running dataset utils')
    dataset_csv =   '/home/valerie/Projects/Alps_LCC/data/split_subset/val_subset.csv'
   # dataset_csv = '/home/valerie/Projects/Alps_LCC/data/split_subset/train_subset.csv'
    
    
    img_dir = '/home/valerie/data/rocky_tlm/rgb/'  #'/data/xiaolong/rgb'
    dem_dir = '/home/valerie/data/rocky_tlm/dem/' # /data/xiaolong/dem'
    mask_dir = '/home/valerie/data/ace_alps/mask'
    
    common_transform = Compose([
        MyRandomHorizontalFlip(p=0.5),
        MyRandomVerticalFlip(p=0.5),
        MyRandomRotation90(p=0.5),
        ])
        
    img_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)
        ]) 
    
    
    
    dataset = SwissImage(dataset_csv=dataset_csv,img_dir=img_dir,dem_dir=dem_dir, 
                         label_dir=mask_dir, common_transform=common_transform,
                         img_transform=img_transform)
    print('len dataset',len(dataset))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=104, shuffle=True)
    
    
    with torch.no_grad():
        run_mean = torch.tensor([0.,])
        run_stds = torch.tensor([0.,])
        
        for batch in tqdm(dataloader):
            
            mean = torch.mean(batch[1],dim=[0,2,3])
            stds = torch.std( batch[1],dim=[2,3]).mean()
            
            run_mean += mean 
            run_stds += stds
        
        ave_mean = run_mean/len(dataloader)
        ave_std =  run_stds/len(dataloader)
                
        print('-'*20,'\n','batch count : ',len(dataset))
        print('DEM: ave_mean',ave_mean, 'ave_std',ave_std)
    
      
    with torch.no_grad():
        
        run_mean = torch.tensor([0.,0.,0.])
        run_stds = torch.tensor([0.,0.,0.])
        for batch in tqdm(dataloader):
            
            mean = torch.mean(batch[0],dim=[0,2,3],)
            stds = torch.std( batch[0],dim=[0,2,3])
            
            run_mean += mean 
            run_stds += stds
        
        ave_mean =  torch.tensor(run_mean)/len(dataloader)
        ave_std =  torch.tensor(run_stds)/len(dataloader)
                
        print('-'*20,'\n','Final count : ',len(dataset))
        print('RGB : ave_mean',ave_mean, 'ave_std',ave_std)
        
    
        
if __name__ =="__main__":
    
    print('running dataset utils')
    compute_mean_std(data_dir=None, full=True)
