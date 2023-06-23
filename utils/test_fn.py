import time
import os
import numpy as np
import torch
import torch.nn as nn
import wandb
import csv
from numpy import linalg as LA
""" 
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
            if isinstance(output, dict):
                output=output['out']
            preds = torch.argmax(output.detach().cpu(),axis=1)
            gt = mask.squeeze().detach().cpu().numpy()
            metrics.update(gt, preds.numpy())
                        
            # measure elapsed time
            tick= time.time()
            
           

                           
        mean_acc, mean_iou, acc_cls, overall_acc = metrics.get_scores()
        classes = {'Background':0, "Bedrock" : 1, "Bedrockwith grass" : 2,
                    "Large blocks" : 3, "Large blocks with grass" : 4, "Scree" : 5,
                    "Scree with grass" : 6,"Water" : 7,
                    "Forest" : 8, "Glacier" : 9, } 
        class_accuracies = { cls : np.round (value,3) for cls, value in zip (classes.keys(),acc_cls )  }
        
        metrics = {
                    'test_macc': mean_acc, 
                    'test_miou' : mean_iou, 
                    'test_acc' : class_accuracies,
                    'test_oacc':overall_acc,   
            }
        if args.log_wandb :
            wandb.log(metrics)
        
        
        
        
        pprint(metrics)
        write_result_to_csv(metrics,args)
        
        
        
 """   
        
def write_result_to_csv(data,args=None):
    import datetime
    time_string = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if args is not None : 
        filename = os.path.join( args.out_dir, args.name,'results.csv')
    else :
        filename = os.path.join('results.csv')
        
    keys = data.keys()
    
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Experiment results'])
        writer.writerow([time_string])
        for key in keys :
            
            if isinstance(data[key],dict):
               continue
            else :
                text = str(key) + str(data[key])
                writer.writerow([text])

    for key in keys :
        if isinstance(data[key],dict):
            sub_data = data[key]
            with open(filename, 'a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=sub_data.keys())
                writer.writeheader()
                writer.writerow(sub_data)
                
        

    print('Results save to file :',filename)
    

if __name__ == '__main__':
    data = {'test_macc': 0.10753172081147054, 'test_miou': 0.014824913622472239, 
            'test_acc': 
                {'Background': 0.303, 'Bedrock': 0.004, 'Bedrockwith grass': 0.0, 
                 'Large blocks': 0.0, 'Large blocks with grass': 0, 
                 'Scree': 0.06, 'Scree with grass': 0.518, 'Water': 0.0, 'Forest': 0.051, }, 
                'test_oacc': 0.0456082
                }
    write_result_to_csv(data)
    