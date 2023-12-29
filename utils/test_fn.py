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

def test_ACE( model,  args):  
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
    device = args.device
    print('Start model testing....')
    test_loader = get_dataloader(args=args,phase='test')
    load_best_model_weights(model,args)
    
    with torch.no_grad():
        
        tick= time.time()
        metrics = MetricLogger( n_classes=args.num_classes)
        
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
                        
        # measure elapsed time
        tack= time.time()
        print('Elapsed time [s]:',int(tack-tick))
        
        # Compute and save metrics : 
        classes = args.classes   
                           
        mean_acc, mean_iou, acc_cls, overall_acc = metrics.get_scores()
        confusion_matrix = metrics.get_confusion_matrix()
        plot_confusion_matrix(confusion_matrix, classes= list(classes.keys()), save_path= os.path.join( args.out_dir, args.name,'confusion_matrix'), normalize=True)
    
        
        cls_acc = { cls : np.round (np.nan_to_num(value),3) for cls, value in zip (classes.keys(),acc_cls )  }
        freq_cls_acc,common_cls_acc,rare_cls_acc = get_group_acc (cls_acc,args)


        metrics = {
                    'test_miou' : np.round ( mean_iou,3), 
                    'test_macc':  np.round (mean_acc,3), 
                    'test_oacc':  np.round (overall_acc,3),
                    'frequent_cls_acc':np.round(freq_cls_acc,3),
                    'common_cls_acc': np.round(common_cls_acc,3),
                    'rare_cls_acc': np.round(rare_cls_acc,3),
                    'test_acc' :  cls_acc,
            }
        if args.log_wandb :
            wandb.log(metrics,step=0)       
        
        
        pprint(metrics)
        write_result_to_csv(metrics,args)
        
        
        
        
        
def get_group_acc (cls_acc,args):

    if args.ds =='TLM' : 
        freq_cls_acc =   1/4* (cls_acc["Scree"]+ cls_acc["Bedrock"] + 
                               cls_acc["Glacier"] + cls_acc["Forest"])
        common_cls_acc = 1/3* (cls_acc["Scree with grass"]+ cls_acc["Water"]+ 
                               cls_acc["Bedrockwith grass"])
        rare_cls_acc =   1/2* (cls_acc["Large blocks"]+ cls_acc["Large blocks with grass"])
    
    else : #FLAIR
        freq_cls_acc = 1/7 *( cls_acc["building"] + cls_acc["pervious surface"]+ 
                             cls_acc["impervious surface"] + cls_acc["deciduous"]+ 
                             cls_acc["brushwood"]+ cls_acc["herbaceous vegetation"]+
                             cls_acc["agricultural land"])
        common_cls_acc = 1/5* ( cls_acc["bare soil"] + cls_acc["water"] + 
                               cls_acc[ "coniferous"] + cls_acc["vineyard"] + 
                               cls_acc["plowed land"] )
        rare_cls_acc =  1/6 * ( cls_acc["swimming pool"] +cls_acc["snow"] +
                               cls_acc["clear cut"] +cls_acc["mixed"] +  
                                cls_acc["ligneous"] +cls_acc["greenhouse"]  )


    return  freq_cls_acc,common_cls_acc,rare_cls_acc 

        
def write_result_to_csv(data,args=None):
    import datetime
    time_string = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if args is not None :        
        filename = os.path.join( args.out_dir, args.name,'metrics.csv')
            
        if args.zero_nontarget_expert :
            filename = os.path.join( args.out_dir, args.name,'metrics_zero_nontarget_exp.csv')
            
            
    else :
        filename = os.path.join('metrics.csv')
        
    keys = data.keys()
    
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Experiment results'])
        writer.writerow([filename])
        writer.writerow([time_string])
        for key in keys :
            
            if isinstance(data[key],dict):
                sub_data = data[key]
                for sub_key in list(sub_data.keys()):
                    text = str(key)+'_'+ str(sub_key) +', '+ str(sub_data[sub_key])
                    writer.writerow([text])
               
            else :
                text = str(key) +', '+ str(data[key])
                writer.writerow([text])
               
       

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
    