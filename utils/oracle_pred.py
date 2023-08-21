
import os, sys
sys.path.append("/home/valerie/Projects/Alps_LCC/") 
import torch
from torch.optim.lr_scheduler import StepLR, LambdaLR, ReduceLROnPlateau
from pprint import pprint
from copy import deepcopy

from utils.training_utils import get_dataloader, get_model, get_criterion, set_all_random_seeds, setup_wandb_log, get_optimizer 
from utils.argparser import parse_args


from utils.train_fn import train_ACE
from utils.validate_fn import validate_ACE
from utils.test_fn import test_ACE
import time
import os
import numpy as np
import torch
import torch.nn as nn
import wandb
import csv
from numpy import linalg as LA

from utils.inference_utils import MetricLogger, get_predictions_from_logits, load_best_model_weights
from utils.visualize import plot_confusion_matrix
from tqdm import tqdm
from utils.training_utils import get_dataloader
from pprint import pprint





def main(args):
    '''run test and store prediction from best model head (3 experts)
    
    Args:
        train_loader (torch.utils.data.DataLoader): dataloader for training set.
        model (torch.nn.Module): image segmentation module.
        device (str): device 'cuda:0' or 'cpu'
        args: arguments from the main script.

    Returns:
        perf_indicator (float): performance indicator: mean IoU over all validation images.
        validation loss (float): val loss
    '''
    print('Start prediction from oracle model....')
    
    # Choose model and device :
    model = get_model(args)
    load_best_model_weights(model,args)
    


    assert args.experts == 3, 'oracle mode only implemented from 3 experts, \nProcess ended' 
    device =  "cuda" if torch.cuda.is_available() or not args.force_cpu else 'cpu'
    model = model.to(device)
    load_best_model_weights(model,args)
    test_loader = get_dataloader(args=args,phase='test')
    
    
    with torch.no_grad():
        
        metrics = MetricLogger( n_classes=10)
        
        # switch to evaluate mode
        model.eval()       
    
        
        for i, (image, dem, mask) in tqdm(enumerate(test_loader)):
            
            # move data to device
            input = torch.cat((image, dem), dim=1).to(device) 
            mask = mask.to(device) 
            num_inputs = input.size(0)
            
            # Run forwards pass and compute loss : 
            output = model(input) 
            
            # Record metrics
            preds = get_predictions_from_logits(output, args)
            gt = mask.squeeze().detach().cpu().numpy()
            metrics.update(gt, preds.numpy())



def save_ACE_as_MCE(args): 
    
    from models.xACE_DeepLabV3Plus import ACE_deeplabv3P_w_Experts
    
    model = ACE_deeplabv3P_w_Experts(10,3)
    
    best_model_path = os.path.join( '/home/valerie/Projects/Alps_LCC/out/baseline/deeplabv3plus_base_all/current_best.pt')
    checkpoint = torch.load(best_model_path)
    state_dict= checkpoint['state_dict']
    layer_list = list(state_dict.keys())
    print('here')
    print('here')






if __name__ == '__main__':
  
    args = parse_args()
    args.device= 'cpu'
    args.model ='MCE'
    args.experts = 3
    args.out_dir='out/trash/tmp_mce2/'
    args.name = 'mce2'
    args.test_only = True
    #save_ACE_as_MCE(args)
    main(args)
        


