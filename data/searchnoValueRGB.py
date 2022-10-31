import os
import sys 
sys.path.append("..") 
import tqdm
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm

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

if __name__ == '__main__':
    data_csv = '/data/xiaolong/master_thesis/data/label_selection_0.1.csv'
    searchnoValue(data_csv)