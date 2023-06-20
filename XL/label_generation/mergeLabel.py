import os
import sys 
sys.path.append("..") 
import tqdm
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm

def mergeLabel(dataset_csv):
    '''
    Merge similar/ not important classes 
    param: csv_file contains the label path
    '''
    mask_dir = '/data/xiaolong/mask'
    tileID = pd.read_csv(dataset_csv)
    for idx in tqdm(range(len(tileID))):
        mask_path = os.path.join(mask_dir, tileID.iloc[idx, 2])
        img = Image.open(mask_path)
        img = np.array(img)
        # merge Feuchtgebiet, Fliessgewaesser, Stehende Gewaesser
        img[(img==6) | (img==13)] = 5
        # merge Gebueschwald, Gehoelzflaeche, Wald, Wald offen
        img[(img==8) | (img==14) | (img==15)] = 7
        # merge Gletscher Schneefeld Toteis
        img[img==12] = 9
        # Lockergestein 10 -> 6
        img[img == 10] = 6
        # Lockergestein locker 11 -> 8 
        img[img == 11] = 8
        img = Image.fromarray(img)
        img.save(mask_path)
        
if __name__ == '__main__':
    data_csv = '/data/xiaolong/master_thesis/data_preprocessing/label_selection_0.1_rgb.csv'
    mergeLabel(data_csv)