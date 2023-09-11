
import os, sys
sys.path.append("/home/valerie/Projects/Alps_LCC/") 
import torch
import time
from pprint import pprint
import os
from tqdm import tqdm
import numpy as np
from utils.training_utils import get_dataloader, get_model 
from utils.argparser import parse_args
from utils.inference_utils import MetricLogger, load_best_model_weights
from utils.training_utils import get_dataloader
import csv
from utils.visualize import plot_confusion_matrix

def write_oracle_result_to_csv(data,args=None,string=None):
    import datetime
    time_string = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if args is not None : 
        filename = os.path.join( args.out_dir, args.name,'oracle_results.csv')
    else :
        filename = os.path.join('oracle_results.csv')
        
    keys = data.keys()
    
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['-'*50])
        writer.writerow(['Experiment results'])
        if string is not None :
            writer.writerow([string])
        writer.writerow(['-'*50])
        writer.writerow([filename])
        writer.writerow([time_string])
        for key in keys :
            
            if isinstance(data[key],dict):
                sub_data = data[key]
                for sub_key in list(sub_data.keys()):
                    text = ' '*8 + str(key)+'_'+ str(sub_key) +', '+ str(sub_data[sub_key])
                    writer.writerow([text])
               
            else :
                text = str(key) +', '+ str(data[key])
                writer.writerow([text])
               
       
    print('Results save to file :',filename)




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
    print('Start prediction as oracle model....')
    assert args.experts == 2 or args.experts ==3, 'oracle mode only implemented from 2-3 experts, \nProcess ended' 
    
    # Choose model and device :
    model = get_model(args)
    load_best_model_weights(model,args)        
    device =  "cuda" if torch.cuda.is_available() or not args.force_cpu else 'cpu'
    model = model.to(device)
    
    
  #  load_best_model_weights(model,args)
    test_loader = get_dataloader(args=args,phase='test')    

        
    
    
    with torch.no_grad():
        
        metrics = MetricLogger( n_classes=10)
        metric_exp0 = MetricLogger( n_classes=10)
        metric_exp1 = MetricLogger( n_classes=10)
        metric_exp2 = MetricLogger( n_classes=10)
        
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
            gt = mask.squeeze().detach().cpu().numpy()
            logits = get_oracle_logits_from_experts(output,mask)
            preds = torch.argmax(logits.detach().cpu(),axis=1)  
            metrics.update(gt, preds.numpy())         
            
            # REcord metrics for each expert
            pred_exp_0 = torch.argmax( output['exp_0'], dim=1)
            metric_exp0.update(gt, pred_exp_0.detach().cpu().numpy())
            
            pred_exp_1 = torch.argmax(  output['exp_1'], dim=1)
            metric_exp1.update(gt, pred_exp_1.detach().cpu().numpy())
            
            if args.experts ==3:
                pred_exp_2 = torch.argmax(  output['exp_2'], dim=1)
                metric_exp2.update(gt, pred_exp_2.detach().cpu().numpy())
            
        
        # measure elapsed time
        tick= time.time()
        print('Elapsed time [s]:',tick)
        
        # Compute and save metrics : 
        classes = {'Background':0, "Bedrock" : 1, "Bedrockwith grass" : 2,
                    "Large blocks" : 3, "Large blocks with grass" : 4, "Scree" : 5,
                    "Scree with grass" : 6,"Water" : 7,
                    "Forest" : 8, "Glacier" : 9, }    
                           
        mean_acc, mean_iou, acc_cls, overall_acc = metrics.get_scores()
 
        class_accuracies = { cls : np.round (value,3) for cls, value in zip (classes.keys(),acc_cls )  }
        freq_cls_acc =   1/4* (class_accuracies["Scree"]+ class_accuracies["Bedrock"] + class_accuracies["Glacier"] + class_accuracies["Forest"])
        common_cls_acc = 1/3* (class_accuracies["Scree with grass"]+    class_accuracies["Water"]+   class_accuracies["Bedrockwith grass"])
        rare_cls_acc =   1/2* (class_accuracies["Large blocks"]+   class_accuracies["Large blocks with grass"])
         
         
        metrics = {
                    'test_miou' : np.round ( mean_iou,3), 
                    'test_macc':  np.round (mean_acc,3), 
                    'test_oacc':  np.round (overall_acc,3),
                    'frequent_cls_acc':np.round(freq_cls_acc,3),
                    'common_cls_acc': np.round(common_cls_acc,3),
                    'rare_cls_acc': np.round(rare_cls_acc,3),
                    'test_acc' :  class_accuracies,
            }
        pprint(metrics)
        write_oracle_result_to_csv(metrics,args)
        
        # Compute and save metrics : 
        print('metrics for expert 0:')
        values = get_score_per_expert(metric_exp0)
        write_oracle_result_to_csv(values,args, 'metrics for expert 0')
        print('*'*50)
        
        print('metrics for expert 1:', )
        values = get_score_per_expert(metric_exp1)
        write_oracle_result_to_csv(values,args, 'metrics for expert 1')
        print('*'*50)
        
        if args.experts ==3:
            print('metrics for expert 2:',)
            values = get_score_per_expert(metric_exp2)
            write_oracle_result_to_csv(values,args, 'metrics for expert 2')
            print('*'*50)
        return


def get_oracle_logits_from_experts(output,gt):
    
    assert isinstance(output,dict)
    nb_experts  = len(output)
    assert nb_experts ==3, 'not implemented for other than 3 experts'
    device = output['exp_0'].device
    
    # Compute one hot encoded expert layer, based on GT label
    body_index = torch.tensor([2,  6, 7]).to(device)
    body_mask =torch.isin(gt,body_index) .float()
    tail_index = torch.tensor([3, 4]).to(device)
    tail_mask = torch.isin(gt, tail_index).float()
    oracle_expert = body_mask + tail_mask*2
    oracle_expert_onehot = torch.nn.functional.one_hot(oracle_expert.long(),nb_experts).moveaxis(-1,0)
    
    exp_logits = torch.stack([output['exp_0'],output['exp_1'],output['exp_2']])
    
    logits =  exp_logits * oracle_expert_onehot
     
    return logits .sum(0)


def get_score_per_expert(metrics):
            
    print('get metrics per expert')
    
    classes = {'Background':0, "Bedrock" : 1, "Bedrockwith grass" : 2,
                    "Large blocks" : 3, "Large blocks with grass" : 4, "Scree" : 5,
                    "Scree with grass" : 6,"Water" : 7,
                    "Forest" : 8, "Glacier" : 9, }
    

    mean_acc, mean_iou, acc_cls, overall_acc = metrics.get_scores()

    
    class_accuracies = { cls : np.round (value,3) for cls, value in zip (classes.keys(),acc_cls )  }
    freq_cls_acc =   1/4* (class_accuracies["Scree"]+ class_accuracies["Bedrock"] + class_accuracies["Glacier"] + class_accuracies["Forest"])
    common_cls_acc = 1/3* (class_accuracies["Scree with grass"]+    class_accuracies["Water"]+   class_accuracies["Bedrockwith grass"])
    rare_cls_acc =   1/2* (class_accuracies["Large blocks"]+   class_accuracies["Large blocks with grass"])
        
    values= {
                'test_miou' : np.round ( mean_iou,3), 
                'test_macc':  np.round (mean_acc,3), 
                'test_oacc':  np.round (overall_acc,3),
                'frequent_cls_acc':np.round(freq_cls_acc,3),
                'common_cls_acc': np.round(common_cls_acc,3),
                'rare_cls_acc': np.round(rare_cls_acc,3),
                'test_acc' :  class_accuracies,
            }
    pprint(values)
    
    return values




if __name__ == '__xmain__':
    size= 8,10,5,5
    
    logits = torch.rand(size=size)
    gt = torch.randint(high =10,size=[8,5,5],  )
    output = {'exp_0': logits, 'exp_1':logits, 'exp_2':logits}
    get_oracle_logits_from_experts(output,gt)
    
    


if __name__ == '__main__':
  
    
    
    
    
    
    args = parse_args()

    args.model ='MCE'
    args.experts = 3
    args.out_dir='out/best/'
    args.name = 'mce_lcom_ws13'
    args.test_only = True
    main(args)
        


