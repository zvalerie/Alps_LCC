import time
import os
import numpy as np
import torch
import wandb
import torch.nn as nn
from numpy import linalg as LA
from utils.training_utils import AverageMeter



def train_ACE(train_loader,  model, criterion, optimizer, epoch, args, device ):
    '''Train one epoch
    
    Args:
        train_loader (torch.utils.data.DataLoader): dataloader for training set.
        model (torch.nn.Module): image segmentation module.
        criterion (torch.nn.Module): loss function for image segmentation.
        optimizer (torch.optim.Optimizer): optimizer for model parameters.
        epoch (int): current training epoch.
        device (str): device 'cuda:0' or 'cpu'
        args: arguments from the main script.
    '''
    
    data_time = AverageMeter()
    losses = AverageMeter()
    mediumlosses = AverageMeter()
    fewlosses = AverageMeter()
    batch_time = AverageMeter()
    
    # switch to train mode
    model.train()
    device = torch.device("cuda")
    end = time.time()
    
    for i, (image, dem, mask) in enumerate(train_loader):
        # measure the data loading time
        data_time.update(time.time() - end)
        
        # move data to device
        input = torch.cat((image, dem), dim=1).to(device)
        mask = mask.long().squeeze().to(device)
        
        # Run forward pass : 
        output = model(input) 
        
        # compute loss
        if args.experts ==0:
            loss = criterion(output,mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                
        elif args.experts == 3:
            raise NotImplementedError
            [many_loss, medium_loss, few_loss], comloss = criterion(output, mask)
            loss = many_loss + medium_loss * args.mediumloss_factor + few_loss * args.fewloss_factor          
            # compute gradient and update
            optimizer.zero_grad()
            # if train seperately:
            if args.train_sep:
                many_loss.backward(retain_graph=True)
                for name, param in model.named_parameters():
                    if not ('medium' in name or 'few' in name):
                        param.requires_grad = False
                
                (medium_loss+few_loss).backward()
                for name, param in model.named_parameters():
                    param.requires_grad = True
            else:
                loss.backward()
            optimizer.step()
            if few_loss != 0:
                fewlosses.update(few_loss.item(), input.size(0))
            if medium_loss != 0:  
                mediumlosses.update(medium_loss.item(), input.size(0))
        
        elif args.experts == 2:
            raise NotImplementedError
            medium_loss = 0
            [many_loss, few_loss], comloss = criterion(output, mask)
            loss = many_loss + few_loss * args.fewloss_factor
            # compute gradient and update
            # if train seperately:
            if args.train_sep:
                optimizer[0].zero_grad()
                optimizer[1].zero_grad()
                many_loss.backward()
                optimizer[0].step()
                few_loss.backward()
                optimizer[1].step()
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            if few_loss != 0:
                fewlosses.update(few_loss.item(), input.size(0))
            if medium_loss != 0:  
                mediumlosses.update(medium_loss.item(), input.size(0))
        
        # update running metrics : 
        losses.update(loss.item(), input.size(0))
        
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.logging_frequency == 0:
            lr = optimizer.param_groups[0]['lr']
       
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                'Speed {speed:.1f} samples/s\t' \
                'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                'Loss {loss.val:.5f} ({loss.avg:.5f})\t'\
                'Learning rate {lr:.5f}\t'\
                'MediumLoss {mediumloss.val:.5f} ({mediumloss.avg:.5f})\t'\
                'FewLoss {fewloss.val:.5f} ({fewloss.avg:.5f})\t'.format(
                epoch, i+1, 
                len(train_loader), 
                batch_time=batch_time,
                speed=input.size(0)/batch_time.val,
                data_time=data_time, loss=losses, lr=lr, 
                mediumloss=mediumlosses, fewloss=fewlosses)
                
            print(msg)
             
    # End of each epoch          
    if args.log_wandb :
        metrics = {
                'train_loss':losses.avg,
                'train_duration': batch_time.avg,
                'lr':lr,    
                'epoch':epoch,                         
            }
            
        wandb.log(metrics)
    
    
    
    



