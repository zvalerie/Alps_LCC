import time
import os
import json
import numpy as np
import torch
import torch.nn as nn
import sys
sys.path.append("/home/valerie/Projects/Alps_LCC/") 
print(sys.path)

import csv
from numpy import linalg as LA
from training_utils import get_dataloader, get_model
from inference_utils import  get_predictions_from_logits, load_best_model_weights
from tqdm import tqdm
from utils.training_utils import get_dataloader
from dataset.SwissImageDataset import  unnormalize_batch
from pprint import pprint
from argparse import Namespace
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from pprint import pprint

TILE_IDS = ['26403_11058', '26370_11109', '26318_11151', '26366_11119', '26308_11133', '25882_11332', '25847_11298', '25659_11101',
            '25869_11328', '26610_11328', '25881_11333', '26308_11149', '25857_11277', '25819_11343', '25954_10892', '26048_11139',
            '26041_11004', '25757_10998', '25811_10849', '25861_10980', '26080_11014', '26301_10891', '25939_10979'  ] 
    

def plot_predictions(image, dem , mask,preds ):
    
    classes = {'Background':0, "Bedrock" : 1, "Bedrockwith grass" : 2,
                "Large blocks" : 3, "Large blocks with grass" : 4, "Scree" : 5,
                "Scree with grass" : 6,"Water" : 7,
                "Forest" : 8, "Glacier" : 9, }
    
    # Process inputs : 
    img =image.movedim(1,-1).cpu().numpy()
    img = unnormalize_images(img)
    dem = dem.squeeze().cpu().numpy()
    mask = mask.squeeze().long().cpu().numpy()
    preds = preds.squeeze().cpu().numpy()
    
    dest_path = args.out_dir +'/' + args.name +'/preds/' 
    if not os.path.exists(dest_path):
        os. mkdir (dest_path)

    # Loop over the samples and plot them
    colors = ['black', 'tab:grey', 'lightgrey', 'maroon', 'red', 'orange', 'yellow', 'royalblue', 'forestgreen','lightcyan',]
    cmap=ListedColormap(colors)
    for k in  range( image.size(0)):
        
        plt.figure(figsize =[20,5])
        plt.subplot(141)
        plt.imshow(img[k])
        plt.title("RGB Image")
        plt.axis('off')

        plt.subplot(142)
        plt.imshow(dem[k], cmap='viridis')
        plt.title("DEM")
        plt.axis('off')

        plt.subplot(143)
        plt.imshow(mask[k], cmap=cmap,vmin=0,vmax=9)
        plt.title("Label")
        # plt.text(0,220,str(np.unique(mask[k])), fontsize = 10)
        plt.axis('off')
                    
        plt.subplot(144)
        plt.imshow(preds[k], cmap=cmap,vmin=0,vmax=9.5)
        plt.title("Predictions")
        plt.axis('off')
        #plt.colorbar(label = list(  classes.keys()) )
        # plt.text(0,220,str(np.unique(preds[k])), fontsize = 15)
        
        # Colorbar parameters :        
        colors = cmap.colors
        values = list(  classes .values())
        txt_labels = list(  classes.keys())
        patches = [ mpatches.Patch(color=colors[i], label= txt_labels[i] ) for i in values ]
        plt.legend(handles=patches, 
            fontsize='small',
            bbox_to_anchor=(1.05, 1), 
            loc=2, 
            frameon = False,
            borderaxespad=0. )            
            
        plt.savefig(dest_path +str(TILE_IDS[k])+'.png',dpi = 300)
        print('fig saved')
        plt.close()



def predict_and_plot(args,model):
    """ plot some  predictions from the model
    """ 
 
    device = args.device
    model.to(device)
    print('Start plotting some examples....')
    test_loader = get_dataloader(args=args,phase='plot')
    if __name__ != '__main__':
        load_best_model_weights(model,args)
    
    nb_img = len(test_loader.dataset)
 
    
    with torch.no_grad():
        # switch to evaluate mode
        model.eval()         
        for i, (image, dem, mask) in tqdm(enumerate(test_loader)):                
            # move data to device
            input = torch.cat((image, dem), dim=1).to(device) 
            mask = mask.to(device)             
            # Run forwards pass : 
            output = model(input)  
            preds = get_predictions_from_logits(output, args)
            print('predict  one batch')

            if args.aggregation in ['MLP_select','CNN_select']:
                plot_predictions(image,dem,mask,preds,)
            
            else :
                plot_predictions(image,dem,mask,preds,)
            print('plot  one batch')
            if args.debug :
                break

                
def unnormalize_images(images, ):
    # Assuming images is a NumPy array with shape (batch_size, height, width, channels)
    # mean and std should be lists or arrays with length equal to the number of channels
    mean = [0.5585, 0.5771, 0.5543]  
    std = [0.2535, 0.2388, 0.2318] 

    unnormalized_images = np.copy(images)
    for i in range(len(mean)):
        unnormalized_images[:, :, :, i] = (unnormalized_images[:, :, :, i] * std[i]) + mean[i]
    
    return unnormalized_images     
            
    
    
    
    
         
if __name__ == '__main__':
    
    exp_path = '/home/valerie/Projects/Alps_LCC/out/august/debug/'
    config_fp = exp_path + 'config.json'
    checkpoint_path = exp_path + 'last_model.pt'
    
    
    # Open the JSON file and load its contents as a dictionary
    with open(config_fp, 'r') as json_file:
        cfg = json.load(json_file)
        args = Namespace(**cfg)
        args.pretrained_weights=False
        args.force_cpu = True
        args.finetune_classifier_only = False
        args.bs = 2
        args.device = 'cpu'
    checkpoint = torch.load(checkpoint_path)
    model = get_model(args)

    checkpoint = torch.load(checkpoint_path)      
    best_weights = checkpoint['state_dict']
    model.load_state_dict (best_weights)
    
    pprint(args)
    
    predict_and_plot(args,model)
        

  
  