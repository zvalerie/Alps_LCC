import os
import numpy as np
import rasterio
from random import sample
from sys import path as sys_path
sys_path.append('/home/valerie/Projects/Alps_LCC') 
import csv
import torch
from torch.utils.data import Dataset
from dataset.dataset_utils import RGBJitter
import torchvision.transforms as T
from torch.utils.data import DataLoader

## From the FLAIR Repository  : 
#https://github.com/IGNF/FLAIR-1-AI-Challenge/blob/main/py_module/dataset.py


class FLAIRDataset(Dataset):

    def __init__(self,   dataset_csv, data_dir,patch_size=200, phase='test' ):
        
        self.data_dir =data_dir
        self.means = [0.4339, 0.4512, 0.4134, 0.4027,0]
        self.stds =  [0.1368, 0.1213, 0.1181, 0.0958,1]
        self.patch_size = patch_size
        split_path = dataset_csv      

        # Read list of tile id from file:                        
        self.id_list = []
        with open(split_path, 'r') as fp:
            csv_reader = csv.reader(fp, delimiter=',')
            for row in csv_reader:  
                self.id_list.append( row) 
                
        self.all_id_list = self.id_list

        # Print Dataset Description : 
        print(  'Get FLAIR Dataset from', '/'.join(split_path.split('/')[-3:]))
        print( '\tWith',len(self.id_list), 'samples',  
                'from ',split_path.split('/')[-2] ,'split')
        print(  '\tPatch  size : ', self.patch_size )                  

        # Define COLOR TRANSFORM 
        if phase == 'train': 
            # Apply augmentations in train phase :
            print('\tUse data augmentation in train phase')  
            self.augm = T.Compose([ 
                        RGBJitter(0.4) ,  
                        T.Normalize( self.means, self.stds),
                        T.RandomVerticalFlip(p=0.5),
                        T.RandomHorizontalFlip(p=0.5),
                        T.RandomCrop(self.patch_size ),
                    ])              
        else:
            self.augm = T.Compose([   
                        T.Normalize( self.means, self.stds),
                        T.RandomCrop(self.patch_size ),
                    ]) 
        
    def __len__(self):
        return len(self.id_list)
    
    def sample_id_list(self):
       
        if len(self.all_id_list) > 5000 and self.phase == 'train':
            self.id_list = sample(self.all_id_list,5000)
            print('random sampling of', len(self.id_list),
                  'samples instead of', len(self.all_id_list))
    
    def read_img(self, raster_file: str) -> np.ndarray:
        with rasterio.open(raster_file) as src_img:
            array = src_img.read() 
            array = array
            array = torch.from_numpy(array[:4,:,:]/255).float() # do not select additional bands  
            return array

    def read_msk(self, raster_file: str) -> np.ndarray:
        with rasterio.open(raster_file) as src_msk:
            array = src_msk.read()[0]
           # array[array>12]=0 # set background as label
            array[array==19]=0 # remap background as label 0
            return torch.from_numpy (array).float().unsqueeze(0)

    def __getitem__(self, index):

        img_id,msk_id = self.id_list[index]
        img = self.read_img(self.data_dir + img_id )        
        mask =  self.read_msk(self.data_dir + msk_id)

        sample = torch.cat((img,mask),axis=0)        
        sample = self.augm(sample)
        rgb,dem,mask = sample[:3,:,:],sample[3,:,:],sample[4,:,:].long()
        return rgb, dem.unsqueeze(0), mask .unsqueeze(0)
    
    


def compute_mean_std(dataset ):
    run_mean = torch.tensor([0.,0.,0.,0.])
    run_stds = torch.tensor([0.,0.,0.,0.])
    count= 0
    print('compute mean and std')
    for x  in tqdm(dataset):
        rgb,dem,mask  = x
        img =  torch.cat((rgb,dem),axis=0) 
        means = img.mean((1,2))
        stds = img.std((1,2))
        run_mean += means
        run_stds += stds
        count+=1
        if count%500 ==0:
            print('count',count, 'means',run_mean/count, 'stds', run_stds/count,)
            
            
    print('-'*20,'\n','Final count : ')
    print('count',count, 'means',run_mean/count, 'stds', np.array(run_stds)/count,)
    
    


def compute_FLAIR_class_frequencies(dataset):
    import collections

    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True,num_workers=8)

    # Initialize a dictionary to store class frequencies
    class_frequencies = collections.Counter()
    

    # Count class occurrences in the dataset
    for _,_, labels in tqdm (dataloader):
        
            labs = labels.flatten()
            unique_cls,counts = torch.unique(labs, return_counts =True)
            batch_values  = {x.item():y.item() for x,y in zip (unique_cls,counts)}
            class_frequencies.update(batch_values)
    print(class_frequencies)

    return class_frequencies


if __name__ =="__main__":


    
    ds= FLAIRDataset(dataset_csv='data/flair_split/base/train.csv' ,
                    data_dir= '/data/valerie/flair/',
                    patch_size=512,
                    phase='test'  )
    print(len(ds))
    
    from tqdm import tqdm 
    
    # compute_mean_std(ds)
    # Compute class frequencies
    class_frequencies = compute_FLAIR_class_frequencies(ds)

    # Print class frequencies
    for class_label, frequency in class_frequencies.items():
        print(f"Class {class_label}: {frequency} occurrences")
        

    
    if False : 
        for x in tqdm( range( len(ds))):
            #print(x)
            try :
                sample = ds[x]
            except OSError :
                print(x, 'previous tile',sample[2])
                continue

    print('terminated without error')
        
    
    
