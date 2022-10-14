import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

def label_selection(label_path, rgb_path, dem_path, threshold):
    '''Select the labels, rgb and dem imgs with the proportion of background 
    pixels less than the threshold and save them to a csv file'''
    labels_list = []
    rgb_list = []
    dem_list = []
    for filename in tqdm(os.listdir(label_path)):
        try:
            img = Image.open(label_path + '/' + filename)
        except:
            pass
            continue
        img = np.array(img)
        _, counts = np.unique(img, return_counts=True)
        counts = counts / counts.sum()
        # counts[0] = proportion of background pixels
        if (counts[0] <= threshold) :
            prefix = filename.split('_')[0] + '_' + filename.split('_')[1]
            if (os.path.exists(rgb_path + '/' + prefix +'_rgb.tif')):
                rgb_list.append(rgb_path + '/' + prefix +'_rgb.tif')
                dem_list.append(dem_path + '/' + prefix +'_dem.tif')
                labels_list.append(label_path + '/' + filename)
    ls = list(zip(rgb_list, dem_list, labels_list))
    df = pd.DataFrame(data=ls, columns=['rbg', 'dem', 'mask'])
    df.to_csv('label_selction_{}.csv'.format(threshold), index = False)

if __name__ == '__main__':
    rgb_path = '/data/xiaolong/rgb'
    dem_path = '/data/xiaolong/dem'
    mask_path = '/data/xiaolong/mask'
    threshold = 0.1
    label_selection(mask_path, rgb_path, dem_path, threshold)