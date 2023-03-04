import logging
import time
import os

import numpy as np
import torch
import torch.nn as nn

from lib.core.inference import get_final_preds
from lib.utils.vis import vis_seg_mask
from lib.utils.evaluation import MetricLogger
# from lib.utils.evaluation import createConfusionMatrix
from tqdm import tqdm
import pandas as pd

logger = logging.getLogger(__name__)

def train(train_loader, train_dataset, model, criterion, optimizer, epoch, output_dir,
          writer_dict, lr_scheduler, args):
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
    device = torch.device("cuda")
    end = time.time()
    
    for i, (image, dem, mask) in enumerate(train_loader):
        # measure the data loading time
        data_time.update(time.time() - end)
        
        # compute output
        image = image.to(device)
        dem = dem.to(device)
        input = torch.cat((image, dem), dim=1) #[B, 4, 400, 400]
        mask = mask.to(device)#[B,10,200,200]
        
        if args.model=='Deeplabv3_proto':
            output = model(input, mask) #[B,10,200,200]
        else :
            output = model(input)
            
        if isinstance(output, dict):
            output = output['seg']
        
        if args.MLP == True:
            # compute loss based on the selection probability of experts
            if args.experts==2:
                [y_many, y_few], MLP_output = output
                few_mask = (mask >= 2) & (mask <= 4) | (mask == 6) | (mask == 7)
                loss = criterion(MLP_output, few_mask.float())
                # [y_many, y_few], MLP_output = output
            elif args.experts==3:
                [y_many, y_medium, y_few], MLP_output = output
                # output = MLP_output
                few_mask = (mask >= 2) & (mask <= 4)
                few_mask = few_mask.float()
                few_mask[few_mask==1] = 2
                medium_mask = (mask == 6) | (mask == 7)
                medium_mask = medium_mask.float()
                mask = few_mask + medium_mask
                # print(mask)
                loss = criterion(MLP_output, mask)
        else:
        # compute loss
            loss = criterion(output, mask)
        
        # compute gradient and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # record loss
        losses.update(loss.item(), input.size(0))
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        # lr_scheduler.step()
        if i % args.frequent == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses)
            logger.info(msg)
    lr = optimizer.param_groups[0]['lr']
    if writer_dict:
        writer = writer_dict['logger']
        global_steps = writer_dict['train_global_steps']

        writer.add_scalar('train_loss', losses.avg, global_steps)
        writer.add_scalar('learning_rate', lr, global_steps)
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
    metrics = MetricLogger(model.num_classes)
    device = torch.device("cuda")
    
    with torch.no_grad():
        end = time.time()
        for i, (image, dem, mask) in enumerate(val_loader):
            # compute output
            image = image.to(device)
            dem = dem.to(device)
            input = torch.cat((image, dem), dim=1) #[B, 4, 200, 200]
            output = model(input) #[B, 10, 200, 200]
            
            num_inputs = input.size(0)
            
            if args.MLP == True:
                if args.experts==2:
                    [y_many, y_few], MLP_output = output
                elif args.experts==3:
                    [y_many, y_medium, y_few], MLP_output = output
                m = nn.Softmax(dim=1)
                MLP_output = m(MLP_output)
                output = MLP_output[:,:1,:,:] * y_many + MLP_output[:,1:2,:,:] * y_medium + MLP_output[:,2:3,:,:] * y_few
                # output = MLP_output
                
            mask = mask.to(device) #[B, 10, 200, 200]
            loss = criterion(output, mask)
            
            # measure accuracy and record loss
            losses.update(loss.item(), num_inputs)
            preds = get_final_preds(output.detach().cpu().numpy())
            gt = mask.squeeze().detach().cpu().numpy()
            metrics.update(gt, preds)
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % args.frequent == 0:
                msg = 'Validate: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses)

                logger.info(msg)
                           
        mean_acc, mean_iou, acc_cls, overall_acc = metrics.get_scores()
        perf_indicator = mean_iou
        metrics.reset()
        logger.info('Mean IoU score: {:.3f}'.format(mean_iou))
        
        if writer_dict:
            writer = writer_dict['logger']
            global_steps = writer_dict['valid_global_steps']

            writer.add_scalar('valid_loss', losses.avg, global_steps)
            writer.add_scalar('valid_iou_score', mean_iou, global_steps)

            writer_dict['valid_global_steps'] = global_steps + 1
    
    return losses.avg, perf_indicator
            
def test(test_loader, test_dataset, model, output_dir,
             writer_dict, args): 
    '''Test the model
    Returns:
        Confusion Matrix
    '''
    batch_time = AverageMeter()
    
    # switch to evaluate mode
    model.eval()
    metrics = MetricLogger(model.num_classes)
    device = torch.device("cuda")
    
    with torch.no_grad():
        end = time.time()
        for i, (image, dem, mask) in enumerate(test_loader):
            
            # compute output
            image = image.to(device)
            dem = dem.to(device)
            input = torch.cat((image, dem), dim=1) #[B, 4, 200, 200]
            output = model(input)
            
            num_inputs = input.size(0)
            if args.MLP == True:
                if args.experts==2:
                    [y_many, y_few], MLP_output = output
                elif args.experts==3:
                    [y_many, y_medium, y_few], MLP_output = output
                m = nn.Softmax(dim=1)
                MLP_output = m(MLP_output)
                output = MLP_output[:,:1,:,:] * y_many + MLP_output[:,1:2,:,:] * y_medium + MLP_output[:,2:3,:,:] * y_few
                
            mask = mask.to(device)
            
            # measure accuracy
            preds = get_final_preds(output.detach().cpu().numpy())
            # gt = torch.squeeze(mask).detach().cpu().numpy()
            gt = mask.squeeze(0).detach().cpu().numpy()
            metrics.update(gt, preds)

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
                input_image = input_image * test_dataset.std.reshape(3,1,1) + test_dataset.mean.reshape(3,1,1)
                input_image[input_image > 1.0] = 1.0
                input_image[input_image < 0.0] = 0.0

                ## Turn the numerical labels into colorful map
                mask_image = mask.detach().cpu().numpy()[idx].astype(np.int64)
                mask_image = vis_seg_mask(mask_image.squeeze())

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
            
        mean_cls, mean_iou, acc_cls, overall_acc = metrics.get_scores()
        many_acc, medium_acc, few_acc = metrics.get_acc_cat()
        confusionMatrix = metrics.get_confusion_matrix()
        
        metrics.reset()
        
        logger.info('Mean IoU score: {:.3f}'.format(mean_iou))
        logger.info('Mean accuracy: {:.3f}'.format(mean_cls))
        logger.info('Overall accuracy: {:.3f}'.format(overall_acc))
        classes = ["Background","Bedrock", "Bedrock with grass", "Large blocks", "Large blocks with grass", 
         "Scree", "Scree with grass", "Water area", "Forest", "Glacier"]
        for i in range(len(acc_cls)):
            logger.info(classes[i] + 'accuracy: {:.3f}'.format(acc_cls[i]))
        
        logger.info('Many accuracy: {:.3f}'.format(many_acc))
        logger.info('Medium accuracy: {:.3f}'.format(medium_acc))
        logger.info('Few accuracy: {:.3f}'.format(few_acc))
        
    return confusionMatrix

def ratio_acc_test(test_loader, test_dataset, model, output_dir,
             writer_dict, args): 
    '''
    Return the proportion of each class in an image and its corresponding accuracy
    '''
    batch_time = AverageMeter()
    
    # switch to evaluate mode
    model.eval()
    metrics = MetricLogger(model.num_classes)
    device = torch.device("cuda")
    
    with torch.no_grad():
        class_id = []
        ratio = []
        accuracy = []
        for i, (image, dem, mask) in tqdm(enumerate(test_loader)):
            
            # compute output
            image = image.to(device)
            dem = dem.to(device)
            input = torch.cat((image, dem), dim=1) #[B, 4, 200, 200]
            output = model(input)
            
            if args.MLP == True:
                # direct output
                _, MLP_output= output
                output = MLP_output
                
                ## selection probability
                # [y_many, y_few], MLP_output= output
                # m = nn.Softmax(dim=1)
                # MLP_output = m(MLP_output)
                # output = (MLP_output[:,:1,:,:] * y_many + MLP_output[:,1:2,:,:] * y_few) / 2
                
            mask = mask.to(device)
            
            # measure accuracy
            preds = get_final_preds(output.detach().cpu().numpy())
            # gt = torch.squeeze(mask).detach().cpu().numpy()
            gt = mask.squeeze(0).detach().cpu().numpy()
            metrics.update(gt, preds)
            classes, counts = np.unique(gt, return_counts=True)
            counts = counts / counts.sum()
            classes = classes.astype(int)
            acc = metrics.get_class_acc(classes)
            
            class_id.extend(classes)
            ratio.extend(counts)
            accuracy.extend(acc)

            metrics.reset()
        dict = {'class': class_id, 'ratio':ratio, 'accuracy':accuracy}
        df = pd.DataFrame(dict)
        df.to_csv('/data/xiaolong/master_thesis/ratio_acc.csv',index=False)
        
    return 

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