import time
import numpy as np
import torch
import wandb

from utils.inference_utils import get_predictions_from_logits, MetricLogger, AverageMeter

def validate_ACE(val_loader, model, criterion, epoch, args):  
    '''run validation
    
    Args:
        train_loader (torch.utils.data.DataLoader): dataloader for training set.
        model (torch.nn.Module): image segmentation module.
        criterion (torch.nn.Module): loss function for image segmentation.
        epoch : current epoch 
        args: arguments from the main script.

    Returns:
        perf_indicator (float): performance indicator: mean IoU over all validation images.
        validation loss (float): val loss
    '''
    with torch.no_grad():
        
        tick= time.time()
        batch_time = AverageMeter()
        losses = AverageMeter()
        mediumlosses = AverageMeter()
        fewlosses = AverageMeter()
        metrics = MetricLogger(model.num_classes)
        device  = args.device
        
        # switch to evaluate mode
        model.eval()       
    
        
        for i, (image, dem, mask) in enumerate(val_loader):
            
            # move data to device
            input = torch.cat((image, dem), dim=1).to(device) 
            mask = mask.long().squeeze().to(device)
            num_inputs = input.size(0)
            
            # Run forwards pass and compute loss : 
            output = model(input) 
            loss = criterion(output, mask)
            
            if 'MLP' in args.aggregation or 'CNN' in args.aggregation :
                # nothing special to do here, just pass
                pass 
                
            elif args.experts == 2 :    
                fewlosses.update(loss[1].item(), input.size(0))           
                loss = loss[0]+loss[1]
                
            elif args.experts == 3 :
                fewlosses.update(loss[2].item(), input.size(0))
                mediumlosses.update(loss[1].item(), input.size(0))
                loss = loss[0] + loss[1] + loss [2]
                
                
            
            # Record metrics
            losses.update(loss.item(), num_inputs)
            

            preds = get_predictions_from_logits(output, args)
            

            gt = mask.squeeze().detach().cpu().numpy()
            metrics.update(gt, preds.detach().cpu().numpy())
                        
        # measure elapsed time
        batch_time.update(time.time() - tick)            
        mean_acc, mean_iou, acc_cls, overall_acc = metrics.get_scores()   
        
        msg = 'Validate: [{0}]\t' \
                'Time {batch_time.avg:.3f}s \t' \
                'Loss {loss.avg:.3f} \t'\
                'MediumLoss {mediumloss.avg:.5f}\t'\
                'FewLoss {fewloss.avg:.5f}\t'\
                'Mean Accuracy {mean_acc:.3f} \t' \
                'Mean IoU {mean_iou:.3f} \t' \
                'Overall Acc {overall_acc:.3f} \t'.format(    
                epoch,              
                batch_time=batch_time,
                mean_acc=mean_acc,
                mean_iou=mean_iou,
                overall_acc=overall_acc,                
                loss=losses,
                mediumloss=mediumlosses, 
                fewloss=fewlosses
                )

        print(msg)
                           
        
        perf_indicator = mean_acc
        metrics.reset()
        
        if args.log_wandb :        
            
            classes = {'Background':0, "Bedrock" : 1, "Bedrockwith grass" : 2,
                    "Large blocks" : 3, "Large blocks with grass" : 4, "Scree" : 5,
                    "Scree with grass" : 6,"Water" : 7,
                    "Forest" : 8, "Glacier" : 9, }  
            class_accuracies = { cls : np.round (value,3) for cls, value in zip (classes.keys(),acc_cls )  }
            
            metrics = {
                    'val_loss':losses.avg,
                    'val_duration': batch_time.avg,
                    'val_macc': mean_acc, 
                    'val_miou' : mean_iou, 
                    'val_acc' : class_accuracies,
                    'val_oacc':overall_acc,   
            }
            wandb.log(metrics)
    
    return losses.avg, perf_indicator