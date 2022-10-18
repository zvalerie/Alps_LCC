import os
from unicodedata import category
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

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


def get_stat(filepath, save_name):
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
    
if __name__ == '__main__':
    rgb_path = '/data/xiaolong/rgb'
    dem_path = '/data/xiaolong/dem'
    mask_path = '/data/xiaolong/mask'
    threshold = 0.1
    label_csv_path = '/data/xiaolong/master_thesis/data/label_selection_0.1.csv'
    counts_path = '/data/xiaolong/master_thesis/data/mainclass_distribution.txt'
    label_selection(mask_path, rgb_path, threshold)
    get_stat(label_csv_path, 'mainclass_distribution')