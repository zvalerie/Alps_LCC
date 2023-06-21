import time
import os
import numpy as np
import torch
import torch.nn as nn
from numpy import linalg as LA
from utils.training_utils import AverageMeter
from XL.lib.utils.evaluation import MetricLogger


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
            preds = torch.argmax(output.detach.cpu.numpy(),axis=1)

            gt = mask.squeeze().detach().cpu().numpy()
            metrics.update(gt, preds)
                        
            # measure elapsed time
            batch_time.update(time.time() - tick)
            tick= time.time()
            
            if i % args.frequent == 0:
                msg = 'Validate: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses)

                print(msg)
                           
        mean_acc, mean_iou, acc_cls, overall_acc = metrics.get_scores()
        perf_indicator = mean_iou
        metrics.reset()
        print('Mean IoU score: {:.3f}'.format(mean_iou))
        
        if False : #writer_dict:
            writer = writer_dict['logger']
            global_steps = writer_dict['valid_global_steps']

            writer.add_scalar('valid_loss', losses.avg, global_steps)
            writer.add_scalar('valid_iou_score', mean_iou, global_steps)

            writer_dict['valid_global_steps'] = global_steps + 1
    
    return losses.avg, perf_indicator