import logging
import time
import os
from turtle import back

import numpy as np
import torch

from inference import get_final_preds
from utils.vis import vis_seg_mask
from utils.evaluation import calc_IoU

logger = logging.getLogger(__name__)

def train(train_loader, model, criterion, optimizer, epoch, output_dir,
          writer_dict, *args):
    '''Train one epoch
    
    Args:
        train_loader (torch.utils.data.DataLoader): dataloader for training set.
        model (torch.nn.Module): image segmentation module.
        criterion (torch.nn.Module): loss function for image segmentation.
        optimizer (torch.optim.Optimizer): optimizer for model parameters.
        epoch (int): current training epoch.
        output_dir (str): directory to save logs.
        writer_dict (dict): dictionary containing tensorboard related objects.
        args: arguments from the main script.
    '''
    
    data_time = AverageMeter()
    losses = AverageMeter()
    batch_time = AverageMeter()
    
    # switch to train mode
    model.train()
    
    end = time.time()
    
    for i, (image, dem, mask) in enumerate(train_loader):
        # measure the data loading time
        data_time.updata(time.time() - end)
        
        # compute output
        input = torch.cat((image, dem), dim=1) #[B, 4, 400, 400]
        output = model(input)
        
        # compute loss
        mask = mask.to(output.device)
        loss = criterion(output, mask)
        
        # compute gradient and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # record loss
        losses.update(loss.item(), input.size(0))
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        
        if i % args.frequency == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses)
            logger.info(msg)
            
            if writer_dict:
                writer = writer_dict['logger']
                global_steps = writer_dict['train_global_steps']

                writer.add_scalar('train_loss', losses.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1
                
                
def validate(val_loader, val_dataset, model, criterion, output_dir,
             writer_dict, args): 
    '''Validate the model
    Returns:
        perf_indicator (float): performance indicator. In the case of image segmentation, we return
                                mean IoU over all validation images.
    '''
    batch_time = AverageMeter()
    losses = AverageMeter()
    
    # switch to evaluate mode
    model.eval()
    
    all_preds = []
    all_gts = []
    
    with torch.no_grad():
        end = time.time()
        for i, (image, dem, mask) in enumerate(val_loader):
            # compute output
            input = torch.cat((image, dem), dim=1) #[B, 4, 400, 400]
            output = model(input)
            
            num_inputs = input.size(0)
            # compute loss
            mask = mask.to(output.device)
            loss = criterion(output, mask)

            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
            preds = get_final_preds(output.detach().cpu().numpy())
            
            all_preds.extend(preds)
            all_gts.extend(mask)
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if writer_dict:
                writer = writer_dict['logger']
                global_steps = writer_dict['vis_global_steps']

                # Pick a random image in the batch to visualize
                idx = np.random.randint(0, num_inputs)

                # Unnormalize the image to [0, 255] to visualize
                input_image = image.detach().cpu().numpy()[idx]
                input_image = input_image * val_dataset.std.squeeze(0) + val_dataset.mean.squeeze(0)
                input_image[input_image > 1.0] = 1.0
                input_image[input_image < 0.0] = 0.0
                input_image = input_image * 255

                ## Turn the numerical labels into colorful map
                mask_image = mask.detach().cpu().numpy()[idx].astype(np.int64)
                mask_image = vis_seg_mask(mask_image)

                output = torch.nn.functional.softmax(output, dim=1)
                output_mask = torch.argmax(output, dim=1, keepdim=False)

                output_mask = output_mask.detach().cpu().numpy()[idx]
                output_mask = vis_seg_mask(output_mask)


                writer.add_image('input_image', input_image, global_steps,
                    dataformats='CHW')
                writer.add_image('result_vis', output_mask, global_steps,
                    dataformats='HWC')
                writer.add_image('gt_mask', mask_image, global_steps,
                    dataformats='HWC')

                writer_dict['vis_global_steps'] = global_steps + 1         

        all_preds = np.concatenate(all_preds, axis=0)
        all_gts = np.concatenate(all_gts, axis=0)
            
        avg_iou_score = calc_IoU(all_preds, all_gts, 16)
        perf_indicator = avg_iou_score
        
        logger.info('Mean IoU score: {:.3f}'.format(avg_iou_score))
        
        if writer_dict:
            writer = writer_dict['logger']
            global_steps = writer_dict['valid_global_steps']

            writer.add_scalar('valid_loss', losses.avg, global_steps)
            writer.add_scalar('valid_iou_score', avg_iou_score, global_steps)

            writer_dict['valid_global_steps'] = global_steps + 1
    
    return perf_indicator
            
def test(test_loader, test_dataset, model, output_dir,
             writer_dict, args): 
    '''Test the model
    Returns:
        perf_indicator (float): performance indicator. In the case of image segmentation, we return
                                mean IoU over all validation images.
    '''
    batch_time = AverageMeter()
    
    # switch to evaluate mode
    model.eval()
    
    all_preds = []
    all_gts = []
    
    with torch.no_grad():
        end = time.time()
        for i, (image, dem, mask) in enumerate(test_loader):
            # compute output
            input = torch.cat((image, dem), dim=1) #[B, 4, 400, 400]
            output = model(input)
            
            num_inputs = input.size(0)
            # compute loss
            mask = mask.to(output.device)
            
            # measure accuracy
            preds = get_final_preds(output.detach().cpu().numpy())
            
            all_preds.extend(preds)
            all_gts.extend(mask)
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if writer_dict:
                writer = writer_dict['logger']
                global_steps = writer_dict['vis_global_steps']

                # Pick a random image in the batch to visualize
                idx = np.random.randint(0, num_inputs)

                # Unnormalize the image to [0, 255] to visualize
                input_image = image.detach().cpu().numpy()[idx]
                input_image = input_image * test_dataset.std.squeeze(0) + test_dataset.mean.squeeze(0)
                input_image[input_image > 1.0] = 1.0
                input_image[input_image < 0.0] = 0.0
                input_image = input_image * 255

                ## Turn the numerical labels into colorful map
                mask_image = mask.detach().cpu().numpy()[idx].astype(np.int64)
                mask_image = vis_seg_mask(mask_image)

                output = torch.nn.functional.softmax(output, dim=1)
                output_mask = torch.argmax(output, dim=1, keepdim=False)

                output_mask = output_mask.detach().cpu().numpy()[idx]
                output_mask = vis_seg_mask(output_mask)

                writer.add_image('input_image', input_image, global_steps,
                    dataformats='CHW')
                writer.add_image('result_vis', output_mask, global_steps,
                    dataformats='HWC')
                writer.add_image('gt_mask', mask_image, global_steps,
                    dataformats='HWC')

                writer_dict['vis_global_steps'] = global_steps + 1         

        all_preds = np.concatenate(all_preds, axis=0)
        all_gts = np.concatenate(all_gts, axis=0)
            
        avg_iou_score = calc_IoU(all_preds, all_gts, 16)
        perf_indicator = avg_iou_score
        
        logger.info('Mean IoU score: {:.3f}'.format(avg_iou_score))
        
        if writer_dict:
            writer = writer_dict['logger']
            global_steps = writer_dict['test_global_steps']
            
            writer.add_scalar('test_iou_score', avg_iou_score, global_steps)

            writer_dict['test_global_steps'] = global_steps + 1
            
    return perf_indicator
   
class AverageMeter(object):
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count