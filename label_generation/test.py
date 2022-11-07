import os
import sys 
sys.path.append("..") 
import tqdm
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm

def labelValue(dataset_csv):
    '''
    show the label value
    '''
    mask_dir = '/data/xiaolong/mask'
    tileID = pd.read_csv(dataset_csv)
    ls = [] 
    for idx in tqdm(range(len(tileID))):
        mask_path = os.path.join(mask_dir, tileID.iloc[idx, 2])
        img = Image.open(mask_path)
        img = np.array(img)
        classes = np.unique(img)
        for Class in classes:
            if Class not in ls:
                ls.append(Class)
    
    print(ls)
    
if __name__ == '__main__':
    data_csv = '/data/xiaolong/master_thesis/data_preprocessing/label_selection_0.1_rgb.csv'
    # labelValue(data_csv)