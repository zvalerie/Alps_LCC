import time
import os
import numpy as np
import torch
import torch.nn as nn
import wandb
import pandas as pd
from numpy import linalg as LA
from utils.training_utils import AverageMeter
from XL.lib.utils.evaluation import MetricLogger
from tqdm import tqdm
from utils.training_utils import get_dataloader
from pprint import pprint

def test_ACE( model,  args, device):  
    '''run validation
    
    Args:
        train_loader (torch.utils.data.DataLoader): dataloader for training set.
        model (torch.nn.Module): image segmentation module.
        device (str): device 'cuda:0' or 'cpu'
        args: arguments from the main script.

    Returns:
        perf_indicator (float): performance indicator: mean IoU over all validation images.
        validation loss (float): val loss
    '''
    print('Start model testing....')
    test_loader = get_dataloader(args=args,phase='test')
    
    with torch.no_grad():
        
        tick= time.time()
        metrics = MetricLogger(model.num_classes)
        
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
            preds = torch.argmax(output.detach().cpu(),axis=1)
            gt = mask.squeeze().detach().cpu().numpy()
            metrics.update(gt, preds.numpy())
                        
            # measure elapsed time
            tick= time.time()
            
           

                           
        mean_acc, mean_iou, acc_cls, overall_acc = metrics.get_scores()
        classes = {'Background':0, "Fels" : 1, "Fels locker" : 2,
                "Felsbloecke" : 3, "Felsbloecke locker" : 4, "Lockergestein" : 5,
                "Lockergestein locker" : 6,"Fliessgewaesser" : 7,
                "Wald" : 8, "Gletscher" : 9, } 
        class_accuracies = { cls : np.round (value,3) for cls, value in zip (classes.keys(),acc_cls )  }
        
        metrics = {'test_mean_acc': mean_acc, 
                'test_mean_iou' : mean_iou, 
                'test_class_accuracies' : class_accuracies,
                'test_overall_acc':overall_acc,   
        }
        if args.log_wandb :
            wandb.log(metrics)
        
        df = pd.DataFrame(metrics)
        df.to_csv(os.path.join( args.out_dir, args.name,'results.csv'))
        
        pprint(metrics)
        
        
        
    
        
        
    
