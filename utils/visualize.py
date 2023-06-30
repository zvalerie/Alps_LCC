
import torch
import matplotlib.pyplot as plt 
import wandb
import os
import numpy as np


from sklearn.metrics import confusion_matrix
from torchvision.utils import make_grid
import pandas as pd
from itertools import islice
import utils
from torch.utils.data import DataLoader
from matplotlib.colors import ListedColormap 
import matplotlib.patches as mpatches


def plot_confusion_matrix (cm, classes, save_path =None, normalize=True,):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """   
    import itertools

    
    confusion_matrix = pd.DataFrame(cm )
    confusion_matrix.to_csv(save_path +'.csv', header=None,index= classes) 


    if normalize:
        norm_cm = np.zeros(shape=[len(classes),len(classes)]) 
        sum_true = cm.sum(axis=1)
        for k in range(len(classes)) : 
            if sum_true[k] ==0:
                norm_cm[k] = np.zeros(len(classes))
            else:   
                norm_cm[k] = cm[k] / sum_true[k] 
        cm =norm_cm
        
        
   
            
    plt.imshow(cm, interpolation='nearest', vmin=0, vmax=1, cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90,fontsize='x-small',)
    plt.yticks(tick_marks, classes,fontsize='x-small',)
    fmt = '.2f' if normalize else '.0f'
    thresh =  0.7

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                fontsize=4 if cm[i, j] >0.01 else 0,
                color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label',fontsize='x-small',)
    plt.xlabel('Predicted label',fontsize='x-small',)
    

    
    plt.savefig( save_path +'.png' ,dpi=300)
    plt.close()
    
    print('Confusion matrix is saved.')
    
    
    return cm
