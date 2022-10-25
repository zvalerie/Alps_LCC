import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
sys.path.append("..") 
from lib.dataset.SwissImage import SwissImage


def label_selection(label_path, rgb_path, threshold):
    '''Select the labels, rgb and dem imgs with the proportion of background 
    pixels less than the threshold and save them to a csv file'''
    labels_list = []
    rgb_list = []
    dem_list = []
    mainClass_list = []
    for filename in tqdm(os.listdir(label_path)):
        try:
            img = Image.open(label_path + '/' + filename)
        except:
            pass
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

def class_distribution(filepath, save_name):
    '''return the number of each class'''
    df = pd.read_csv(filepath)
    mainclasses = df['mainclass'].values
    mainclass, counts = np.unique(mainclasses, return_counts=True)
    stat = np.concatenate((mainclass.reshape(mainclass.size, 1),counts.reshape(counts.size,1)), axis = 1).astype(int)
    np.savetxt("{}.txt".format(save_name), stat, fmt='%i')
    return

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

def subset(csv_list, name_list, frac):
    '''
    use stratified selection to extract certain percent of data as subset
    '''
    for i in range(len(csv_list)):
        dataset = pd.read_csv(csv_list[i])
        subset = dataset.groupby('mainclass').sample(frac = frac, random_state = 1)
        subset.to_csv(name_list[i]+'_subset.csv',index = False)

def getStat(train_data):
    '''
    Compute mean and variance for training data
    param: train_data: Dataset
    return: mean, std
    '''
    print(len(train_data))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _, _ in tqdm(train_loader):
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())

if __name__ == '__main__':
    
    rgb_path = '/data/xiaolong/rgb'
    dem_path = '/data/xiaolong/dem'
    mask_path = '/data/xiaolong/mask'
    threshold = 0.1
    # label_selection(mask_path, rgb_path, threshold)
    
    
    label_csv_path = '/data/xiaolong/master_thesis/data/label_selection_0.1.csv'
    counts_path = '/data/xiaolong/master_thesis/data/mainclass_distribution.txt'
    # class_distribution(label_csv_path, 'mainclass_distribution')
    
    
    csv_list = ['/data/xiaolong/master_thesis/data/train_dataset.csv',
                '/data/xiaolong/master_thesis/data/val_dataset.csv',
                '/data/xiaolong/master_thesis/data/test_dataset.csv']
    name_list = ['train', 'val', 'test']
    # subset(csv_list, name_list, 0.1)
    
    train_csv = '/data/xiaolong/master_thesis/data/subset/train_subset.csv'
    img_dir = '/data/xiaolong/rgb'
    dem_dir = '/data/xiaolong/dem'
    mask_dir = '/data/xiaolong/mask'
    train_dataset = SwissImage(train_csv, img_dir, dem_dir, mask_dir)
    mean, std = getStat(train_dataset)
    np.savetxt("mean.txt", mean, fmt='%.04f')
    np.savetxt("std.txt", std, fmt='%.04f')

