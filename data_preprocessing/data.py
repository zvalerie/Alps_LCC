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

def few_categories_tile_selection(filepath, few_index):
    '''Select the tiles that contains few categories'''
    ls = []
    df = pd.read_csv(filepath)
    mask_dir = '/data/xiaolong/mask'
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
    ls = ['train', 'val', 'test']
    for i in range(3):
        df = pd.read_csv(csv_path[i])
        new_df = pd.DataFrame(columns=['rbg','dem','mask','mainclass'])
        mask_dir = '/data/xiaolong/mask'
        for idx in tqdm(range(len(df))):
            mask_path = os.path.join(mask_dir, df.iloc[idx, 2])
            img = Image.open(mask_path)
            img = np.array(img)
            classes, counts = np.unique(img, return_counts=True)
            inter = [j for j in classes if j == 4]
            if inter:
                new_df = pd.concat([new_df, df.iloc[idx, :].to_frame().T], axis=0, ignore_index=True)
        new_df.to_csv('{}_{}_dataset.csv'.format(class_id, ls[i]),index=False)

def subset(csv_list, name_list, frac):
    '''
    use stratified selection to extract certain percent of data as subset
    '''
    for i in range(len(csv_list)):
        dataset = pd.read_csv(csv_list[i])
        subset = dataset.groupby('mainclass').sample(frac = frac, random_state = 1)
        subset.to_csv('./subset/'+name_list[i]+'_subset.csv',index = False)

def getStat(dataset, channel):
    '''
    Compute mean and variance for training data
    param: train_data: Dataset
    return: mean, std
    '''
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(channel)
    std = torch.zeros(channel)
    for _, X, _ in tqdm(train_loader):
        for d in range(channel):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return list(mean.numpy()), list(std.numpy())

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
    train_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=False, num_workers=6,
    pin_memory=True)
    min_value = 100000
    max_value = 0
    for _, dem, _ in tqdm(train_loader):
        dem = dem.numpy()
        min_value = min(min_value, dem.min())
        max_value = max(max_value, dem.max())
    
    print(max_value, min_value)



def calculate_mean_std(dataset):
    mean = np.array([0.,0.,0.])
    stdTemp = np.array([0.,0.,0.])
    std = np.array([0.,0.,0.])

    train_csv = '/data/xiaolong/master_thesis/data_preprocessing/subset/train_subset.csv'
    img_dir = '/data/xiaolong/rgb'
    dem_dir = '/data/xiaolong/dem'
    mask_dir = '/data/xiaolong/mask'
    train_dataset = SwissImage(train_csv, img_dir, dem_dir, mask_dir)

    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=1, shuffle=False, num_workers=0,
            pin_memory=True)

    for _, X, _ in tqdm(train_loader):
        X = np.array(X)
        for j in range(X.shape[0]):
            mean[j] += np.mean(X[:,j,:,:])

    mean = (mean/len(train_dataset))

    for X, _, _ in tqdm(train_loader):
        X = np.array(X)
        for j in range(X.shape[0]):
            stdTemp[j] += ((X[:,j,:,:] - mean[j])**2).sum()/(X.shape[3]*X.shape[2])

    std = np.sqrt(stdTemp/len(train_dataset))

    print(mean)
    print(std)
    
    
if __name__ == '__main__':
    
    # selection the label with background pixels lower than 10%
    rgb_path = '/data/xiaolong/rgb'
    dem_path = '/data/xiaolong/dem'
    mask_path = '/data/xiaolong/mask'
    threshold = 0.1
    # label_selection(mask_path, rgb_path, threshold)
    
    # generate the main class distribution
    label_csv_path = '/data/xiaolong/master_thesis/data/label_selection_0.1_rgb.csv'
    # class_distribution(label_csv_path, 'mainclass_distribution')
    
    # split the data into train, val, test 
    # data_split(label_csv_path, 0.6, 0.2, 0.2)
    
    csv_list = ['/data/xiaolong/master_thesis/data_preprocessing/train_dataset.csv',
                '/data/xiaolong/master_thesis/data_preprocessing/val_dataset.csv',
                '/data/xiaolong/master_thesis/data_preprocessing/test_dataset.csv']
    # name_list = ['train', 'val', 'test']
    # subset(csv_list, name_list, 0.1)
    
    train_csv = '/data/xiaolong/master_thesis/data_preprocessing/subset/train_subset.csv'
    img_dir = '/data/xiaolong/rgb'
    dem_dir = '/data/xiaolong/dem'
    mask_dir = '/data/xiaolong/mask'
    train_dataset = SwissImage(train_csv, img_dir, dem_dir, mask_dir)
    # get_min_max(train_dataset)
    mean, std = getStat(train_dataset, 1)
    print(mean, std)
    # np.savetxt("mean.txt", mean, fmt='%.04f')
    # np.savetxt("std.txt", std, fmt='%.04f')
    
    # data_csv = '/data/xiaolong/master_thesis/data/label_selection_0.1.csv'
    # searchnoValue(data_csv)
    
    file_path = '/data/xiaolong/master_thesis/data_preprocessing/subset/train_subset.csv'
    few_index = [2, 3, 4, 6, 7]
    # few_categories_tile_selection(file_path, few_index)
    
    # csv_list = ['/data/xiaolong/master_thesis/data_preprocessing/subset/train_subset.csv',
    #             '/data/xiaolong/master_thesis/data_preprocessing/subset/val_subset.csv',
    #             '/data/xiaolong/master_thesis/data_preprocessing/subset/test_subset.csv']
    # creat_few_categories_dataset(csv_list, 4)