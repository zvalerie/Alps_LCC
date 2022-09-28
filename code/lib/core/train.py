import logging
import time
import os

import numpy as np
import torch

logger = logging.getLogger(__name__)

def train_model(train_loader, model, criterion, optimizer, epoch, output_dir,
          writer_dict, *args):
    '''Train one epoch'''
    
    data_time = AverageMeter()
    losses = AverageMeter()
    batch_time = AverageMeter()
    
    # switch to train mode
    model.train()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    end = time.time()
    
    for i, (image, dem, mask) in enumerate(train_loader):
        # measure the data loading time
        data_time.updata(time.time() - end)
        
        # compute output
        input = torch.cat([image, dem], dim=0)
        output = model(input.to(device))
        
        # compute loss
        mask = mask.to(device)
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
            
def validate(val_loader, val_dataset, model, criterion, output_dir,
             writer_dict, args): 
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    
    # switch to evaluate mode
    model.eval()
    
    all_preds = []
    all_gts = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    with torch.no_grad():
        end = time.time()
        for i, (image, dem, mask) in enumerate(val_loader):
            # compute output
            input = torch.cat([image, dem], dim=0)
            output = model(input.to(device))
            
            # compute loss
            mask = mask.to(device)
            loss = criterion(output, mask)

            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
            
            
    

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