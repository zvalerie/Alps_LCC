import time
import os
import numpy as np
import torch
import wandb
import torch.nn as nn
from numpy import linalg as LA
from utils.training_utils import AverageMeter
from XL.lib.utils.evaluation import MetricLogger
from tqdm import tqdm

def validate_ACE(val_loader, model, criterion, epoch, args, device):  
    '''run validation
    
    Args:
        train_loader (torch.utils.data.DataLoader): dataloader for training set.
        model (torch.nn.Module): image segmentation module.
        criterion (torch.nn.Module): loss function for image segmentation.
        epoch : current epoch 
        device (str): device 'cuda:0' or 'cpu'
        args: arguments from the main script.

    Returns:
        perf_indicator (float): performance indicator: mean IoU over all validation images.
        validation loss (float): val loss
    '''
    with torch.no_grad():
        
        tick= time.time()
        batch_time = AverageMeter()
        losses = AverageMeter()
        metrics = MetricLogger(model.num_classes)
        
        # switch to evaluate mode
        model.eval()       
    
        
        for i, (image, dem, mask) in enumerate(val_loader):
            
            # move data to device
            input = torch.cat((image, dem), dim=1).to(device) #[B, 4, 200, 200]
            mask = mask.to(device) #[B, 10, 200, 200]
            num_inputs = input.size(0)
            
            # Run forwards pass and compute loss : 
            output = model(input) #[B, 10, 200, 200]
            loss = criterion(output, mask)
            
            # Record metrics
            losses.update(loss.item(), num_inputs)
            preds = torch.argmax(output.detach().cpu(),axis=1)

            gt = mask.squeeze().detach().cpu().numpy()
            metrics.update(gt, preds.numpy())
                        
            # measure elapsed time
            batch_time.update(time.time() - tick)
            tick= time.time()
            
        mean_acc, mean_iou, acc_cls, overall_acc = metrics.get_scores()   
        
        msg = 'Validate: [{0}/{1}]\t' \
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                'Mean Accuracy {mean_acc:.3f} \t' \
                'Mean IoU {mean_iou:.3f} \t' \
                'Overall Acc {overall_acc:.3f} \t' \
                'Loss {loss.val:.3f} ({loss.avg:.3f})'.format(  
                i, len(val_loader), 
                batch_time=batch_time,
                mean_acc=mean_acc,
                mean_iou=mean_iou,
                overall_acc=overall_acc,                
                loss=losses)

        print(msg)
                           
        
        perf_indicator = mean_iou
        metrics.reset()
        print('Mean IoU score: {:.3f}'.format(mean_iou))
        
        if args.log_wandb :
            metrics = {
                'val_loss':loss.avg,
                'val_duration': batch_time.avg,
                'val_mIoU':mean_iou,
                                            
            }
            
            wandb.log(metrics)
            
        classes = {'Background':0, "Fels" : 1, "Fels locker" : 2,
                "Felsbloecke" : 3, "Felsbloecke locker" : 4, "Lockergestein" : 5,
                "Lockergestein locker" : 6,"Fliessgewaesser" : 7,
                "Wald" : 8, "Gletscher" : 9, } 
        class_accuracies = { cls : np.round (value,3) for cls, value in zip (classes.keys(),acc_cls )  }
        
        metrics = {'mean_acc': mean_acc, 
                'mean_iou' : mean_iou, 
                'class_accuracies' : class_accuracies,
                'overall_acc':overall_acc,   
        }

    
    return losses.avg, perf_indicator