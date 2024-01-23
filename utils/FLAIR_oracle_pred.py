
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
from utils.oracle_pred import write_oracle_result_to_csv


@torch.no_grad()
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
    args.num_classes = 10 if args.ds =='TLM' else 19

    # Choose model and device :
    model = get_model(args)
    load_best_model_weights(model,args)        
    device =  "cuda" if torch.cuda.is_available() or not args.force_cpu else 'cpu'
    model = model.to(device)
    
    test_loader = get_dataloader(args=args,phase='test')       
        
    with torch.no_grad():
        
        metrics = MetricLogger( n_classes=args.num_classes)
        metric_exp0 = MetricLogger( n_classes=args.num_classes)
        metric_exp1 = MetricLogger( n_classes=args.num_classes)
        metric_exp2 = MetricLogger( n_classes=args.num_classes)
        
        # switch to evaluate mode
        model.eval()       
        
        for i, (image, dem, mask) in tqdm(enumerate(test_loader)):
            
            # move data to device
            input = torch.cat((image, dem), dim=1).to(device) 
            
            # Run forwards pass and compute loss : 
            output = model(input)      
            
            # Record metrics
            gt = mask.squeeze().detach().cpu().numpy()            
            logits = get_FLAIR_oracle_logits_from_experts(output,mask)
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
        classes =  {
            0:'Others',
            1: "building",      2: "pervious surface",      3: "impervious surface",    4: "bare soil",
            5: "water",         6: "coniferous",            7: "deciduous",             8: "brushwood",
            9: "vineyard",      10: "herbaceous vegetation",11: "agricultural land",    12: "plowed land",
            13: "swimming pool",        14: "snow",    15: "clear cut",    16: "mixed",
            17: "ligneous",    18: "greenhouse",
        } 
                           
        mean_acc, mean_iou, acc_cls, overall_acc = metrics.get_scores()
        acc_cls = np.nan_to_num(acc_cls)
        cls_acc = { cls : np.round (value,3) for cls, value in zip (classes.values(),acc_cls )  }

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
         
        metrics = {
                    'test_miou' : np.round ( mean_iou,3), 
                    'test_macc':  np.round (mean_acc,3), 
                    'test_oacc':  np.round (overall_acc,3),
                    'frequent_cls_acc':np.round(freq_cls_acc,3),
                    'common_cls_acc': np.round(common_cls_acc,3),
                    'rare_cls_acc': np.round(rare_cls_acc,3),
                    'test_acc' :  cls_acc,
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

@torch.no_grad()
def get_FLAIR_oracle_logits_from_experts(output,gt):

    assert isinstance(output,dict)
    nb_experts  = len(output)
    assert nb_experts ==3, 'not implemented for other than 3 experts'

    # concatenate output from all experts with shape : nb expert x B x n_class x H x W
    exp_logits =  torch.stack([output['exp_0'],output['exp_1'],output['exp_2']])
    device = exp_logits.device
    gt = gt.to(device)

    # Compute one hot encoded expert layer, based on GT label
    tail_index = torch.tensor([13,14,15,16,17,18]).to(device)
    body_index = torch.tensor([4,5,6,9,12]).to(device)
    body_mask =torch.isin(gt,body_index) .float()
    tail_mask = torch.isin(gt, tail_index).float()
    oracle_expert = body_mask + tail_mask*2         # shape is B X 1 x H x W
    oracle_expert_onehot = torch.nn.functional.one_hot(oracle_expert.long(),nb_experts).moveaxis(-1,0)
    
    
    logits =  exp_logits * oracle_expert_onehot
    
    return logits .sum(0)


def get_score_per_expert(metrics):
            
    print('get metrics per expert')
    
    classes =  {
            0:'Others',
            1: "building",      2: "pervious surface",      3: "impervious surface",    4: "bare soil",
            5: "water",         6: "coniferous",            7: "deciduous",             8: "brushwood",
            9: "vineyard",      10: "herbaceous vegetation",11: "agricultural land",    12: "plowed land",
            13: "swimming pool",        14: "snow",    15: "clear cut",    16: "mixed",
            17: "ligneous",    18: "greenhouse",
        } 
    

    mean_acc, mean_iou, acc_cls, overall_acc = metrics.get_scores()
    acc_cls = np.nan_to_num(acc_cls)
    
    cls_acc = { cls : np.round (value,3) for cls, value in zip (classes.values(),acc_cls )  }
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
    values= {
                'test_miou' : np.round ( mean_iou,3), 
                'test_macc':  np.round (mean_acc,3), 
                'test_oacc':  np.round (overall_acc,3),
                'frequent_cls_acc':np.round(freq_cls_acc,3),
                'common_cls_acc': np.round(common_cls_acc,3),
                'rare_cls_acc': np.round(rare_cls_acc,3),
                'test_acc' :  cls_acc,
            }
    pprint(values)
    
    return values




    


if __name__ == '__main__':
  
    args = parse_args()
    args.ds ='FLAIR'
    args.backbone ='deeplab'
    args.name = 'FLAIR_dlv3_MCE-3'
    args.lws = False
    args.experts = 3
    args.out_dir='/home/valerie/Projects/Alps_LCC/out/revision/'
    args.large_dataset =True
    args.num_classes = 10 if args.ds =='TLM' else 19
    args.bs =5

    args.test_only = True
    main(args)
        


