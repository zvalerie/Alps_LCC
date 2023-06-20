import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
sys.path.append("..") 
sys.path.append(".") 
from XL.lib.dataset.SwissImage import SwissImage


def label_selection(label_path, rgb_path, threshold):
    '''Select the labels, rgb and dem imgs with the proportion of background 
    pixels less than the threshold and save them to a csv file; find the main class of each tile
    adnd add it in the last column of the csv file'''
    labels_list = []
    rgb_list = []
    dem_list = []
    excepted_img = 0
    mainClass_list = []
    print('Number of file in folder ',label_path,':', len(os.listdir(label_path)))
    print('Number of file in folder ',rgb_path,':', len(os.listdir(label_path)))
    for filename in tqdm(os.listdir(label_path)):
        try:
            img = Image.open(label_path + '/' + filename)
        except:
            excepted_img +=1
            continue
        img = np.array(img)
        classes, counts = np.unique(img, return_counts=True)
        counts = counts / counts.sum()
        mainclass = classes[np.argmax(counts)]
        # counts[0] = proportion of background pixels
        if (classes[0] != 0 or counts[0] <= threshold) :
            prefix = filename.split('_')[0] + '_' + filename.split('_')[1]
            ## skip the images that cannot be opened
            if (os.path.exists(rgb_path + '/' + prefix +'_rgb.tif')):
                rgb_list.append(prefix +'_rgb.tif')
                dem_list.append(prefix +'_dem.tif')
                labels_list.append(filename)
                mainClass_list.append(mainclass)
        
    ls = list(zip(rgb_list, dem_list, labels_list, mainClass_list))
    df = pd.DataFrame(data=ls, columns=['rbg', 'dem', 'mask','mainclass'])
    df.to_csv('label_selection_{}.csv'.format(threshold), index = False)
    print('Number of selected file (tiles above the threshold of',threshold,'):', len(df))
    print('Fraction : ',len(df)/len(os.listdir(label_path)) *100)
    print('Excepted files (canont read):', excepted_img)
    

def class_distribution(filepath, save_name):
    '''return the number of each class'''
    df = pd.read_csv(filepath)
    mainclasses = df['mainclass'].values
    mainclass, counts = np.unique(mainclasses, return_counts=True)
    stat = np.concatenate((mainclass.reshape(mainclass.size, 1),counts.reshape(counts.size,1)), axis = 1).astype(int)
    np.savetxt("{}.txt".format(save_name), stat, fmt='%i')
    return

def few_categories_tile_selection(filepath, few_index):
    '''Select the tiles that contains few categories'''
    ls = []
    df = pd.read_csv(filepath)
    mask_dir = '/home/valerie/data/ace_alps/mask' #'/data/xiaolong/mask'
    for idx in tqdm(range(len(df))):
        mask_path = os.path.join(mask_dir, df.iloc[idx, 2])
        img = Image.open(mask_path)
        img = np.array(img)
        classes, counts = np.unique(img, return_counts=True)
        inter = [i for i in classes if i in few_index]
        if inter:
            ls.append(1)
        else:
            ls.append(0)
    df['few'] = ls
    df.to_csv('train_subset_few.csv',index=False)
    
def data_split(csv_path, train_ratio, val_ratio, test_ratio):
    '''split the dataset into train, val, test set using stratified selection'''
    # read the csv file that contains the tile_id and main class
    df = pd.read_csv(csv_path) 
    #split train+val dataset and test dataset
    train_valset, testset = train_test_split(df, test_size=test_ratio, random_state=1, stratify=df['mainclass'])
    #split train and val dataset
    val_size = val_ratio/(train_ratio + val_ratio)
    trainset, valset = train_test_split(train_valset, test_size=val_size, random_state=1, stratify=train_valset['mainclass'])
    # save to csv
    trainset.to_csv('train_dataset.csv', index = False)
    valset.to_csv('val_dataset.csv', index = False)
    testset.to_csv('test_dataset.csv', index = False)

def creat_few_categories_dataset(csv_path, class_id):
    '''create a dataset that contains only few categories'''
    ls = ['train', 'val', 'test']
    for i in range(3):
        cnt = 0
        df = pd.read_csv(csv_path[i])
        new_df = pd.DataFrame(columns=['rbg','dem','mask','mainclass'])
        mask_dir = '/home/valerie/data/ace_alps/mask' #'/data/xiaolong/mask'
        for idx in tqdm(range(len(df))):
            mask_path = os.path.join(mask_dir, df.iloc[idx, 2])
            img = Image.open(mask_path)
            img = np.array(img)
            classes, counts = np.unique(img, return_counts=True)
            inter = [j for j in classes if j == 4]
            if inter:
                cnt += 1
                new_df = pd.concat([new_df, df.iloc[idx, :].to_frame().T], axis=0, ignore_index=True)
        new_df.to_csv('{}_{}_dataset.csv'.format(class_id, ls[i]),index=False)
    return

def subset(csv_list, name_list, frac):
    '''
    use stratified selection to extract certain percent of data as subset
    '''
    for i in range(len(csv_list)):
        dataset = pd.read_csv(csv_list[i])
        subset = dataset.groupby('mainclass').sample(frac = frac, random_state = 1)
        subset.to_csv('./subset/'+name_list[i]+'_subset.csv',index = False)
    return

def searchnoValue(dataset_csv):
    '''
    search RGB imgs only cantains 0 value
    param: csv_file contains the rgb path
    '''
    ls = []
    rgb_dir = '/data/xiaolong/rgb'
    tileID = pd.read_csv(dataset_csv)
    for idx in tqdm(range(len(tileID))):
        rgb_path = os.path.join(rgb_dir, tileID.iloc[idx, 0])
        img = Image.open(rgb_path)
        img = np.array(img)
        if np.all(img == 0):
            ls.append([idx, tileID.iloc[idx, 0]])
    dropIdx = pd.DataFrame(ls) 
    drop_ls = []
    for i in range(len(dropIdx)):
        drop_ls.append((dropIdx.iloc[i,0]))
    rgb_label_selection = tileID.drop(drop_ls)
    rgb_label_selection.to_csv('label_selection_0.1_rgb.csv',index=False)

def get_min_max(dataset):
    '''get the min and max value of the dem'''
    train_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=False, num_workers=6,
    pin_memory=True)
    min_value = 100000
    max_value = 0
    for _, dem, _ in tqdm(train_loader):
        dem = dem.numpy()
        min_value = min(min_value, dem.min())
        max_value = max(max_value, dem.max())
    
    return min_value, max_value

def cal_mean_std(dataset =None):
    '''calculate the mean and std of the dataset'''
    mean = np.array([0.,0.,0.])
    stdTemp = np.array([0.,0.,0.])
    std = np.array([0.,0.,0.])
    if dataset is None :
        img_dir = '/home/valerie/data/rocky_tlm/rgb/'  #'/data/xiaolong/rgb'
        dem_dir = '/home/valerie/data/rocky_tlm/dem/' # /data/xiaolong/dem'
        mask_dir = '/home/valerie/data/ace_alps/mask'
        ds_csv = '/home/valerie/Projects/Alps_LCC/data/split_subset/train_subset.csv'
        dataset = SwissImage(ds_csv, img_dir, dem_dir, mask_dir)

    train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=0,
            pin_memory=True)

    for X, _, _ in tqdm(train_loader):
        X = np.array(X)
        for j in range(X.shape[0]):
            mean[j] += np.mean(X[:,j,:,:])

    mean = (mean/len(dataset))

    for X, _, _ in tqdm(train_loader):
        X = np.array(X)
        for j in range(X.shape[0]):
            stdTemp[j] += ((X[:,j,:,:] - mean[j])**2).sum()/(X.shape[3]*X.shape[2])
            

    std = np.sqrt(stdTemp/len(dataset)) ###############################################3
    return mean, std

def compute_mean_std(dataset):
    # Initialize variables to store cumulative sum of pixel values
    cumulative_sum = np.zeros(4)
    cumulative_std = np.zeros(4)
    num_batch = 0

    # Iterate through the dataset in batches to calculate cumulative sum
    for img,dem,mask in tqdm(dataset):
        
        # Convert batch of images to NumPy array if needed
        if not isinstance(img, np.ndarray):
            img = np.array(img)
            dem = np.array(dem)

        batch = np.concatenate((img,dem),axis=0)
        
        # Compute sum and sum of squares for the current batch
        cumulative_sum += np.sum(batch, axis=(1, 2))
        cumulative_std += np.std(batch, axis=( 1, 2))
        num_batch += 1

    # Calculate mean and std
    mean = cumulative_sum / num_batch   
    std = cumulative_std / num_batch 
   

    return mean, std


def cal_pixel_frequency(csv_path):
    
    '''return the pixel frequency of each class'''
    df = pd.read_csv(csv_path)
    array = np.zeros((10))
    mask_dir = '/data/xiaolong/mask'
    for idx in tqdm(range(len(df))):
        mask_path = os.path.join(mask_dir, df.iloc[idx, 2])
        img = Image.open(mask_path)
        img = np.array(img)
        classes, counts = np.unique(img, return_counts=True)
        for i in range(len(classes)):
            array[int(classes[i])] += counts[i]
    return array


if __name__ == '__main__':
    
    dataset_csv =   '/home/valerie/Projects/Alps_LCC/data/split_subset/train_subset_few.csv'
        
    img_dir = '/home/valerie/data/rocky_tlm/rgb/'  #'/data/xiaolong/rgb'
    dem_dir = '/home/valerie/data/rocky_tlm/dem/' # /data/xiaolong/dem'
    mask_dir = '/home/valerie/data/ace_alps/mask'
    
    ds = SwissImage(dataset_csv,img_dir,dem_dir,mask_dir,common_transform=None,img_transform=None,debug=False)
    compute_mean_std(ds)