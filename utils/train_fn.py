import time
import os
import numpy as np
import torch
import wandb
import torch.nn as nn
from numpy import linalg as LA
from utils.inference_utils import AverageMeter
from tqdm import tqdm


def freeze_backbone(model):
    
    for name, param in model.named_parameters():
        if not ('medium' in name or 'few' in name):
            param.requires_grad = False
        
def unfreeze_backbone(model) :
    
    for name, param in model.named_parameters():
        param.requires_grad = True
            


def train_ACE(train_loader,  model, criterion, optimizer, epoch, args):
    '''Train one epoch
    
    Args:
        train_loader (torch.utils.data.DataLoader): dataloader for training set.
        model (torch.nn.Module): image segmentation module.
        criterion (torch.nn.Module): loss function for image segmentation.
        optimizer (torch.optim.Optimizer): optimizer for model parameters.
        epoch (int): current training epoch.
        args: arguments from the main script.
    '''
    
 
    losses = AverageMeter()
    mediumlosses = AverageMeter()
    fewlosses = AverageMeter()
    batch_time = AverageMeter()
    
    # switch to train mode
    model.train()
    device = args.device
    end = time.time()
    
    for i, (image, dem, mask) in tqdm(enumerate(train_loader)):
        
        # move data to device
        input = torch.cat((image, dem), dim=1).to(device)
        mask = mask.long().squeeze().to(device)
        
        # Run forward pass : 

        output = model(input) 
        
        # compute loss
        if args.experts ==0 or 'MLP' in args.aggregation or 'CNN' in args.aggregation :
            loss = criterion(output,mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                    
            
        elif args.experts == 2:
            
            many_loss , few_loss = criterion(output, mask)
            loss = many_loss + few_loss 
            
            if not args.separate_backprop or few_loss.item()==0.:
                optimizer.zero_grad() 
                loss.backward()
                optimizer.step()
            
            else : 
                # update backbone with expert 1 and head for few classes with expert few
                optimizer.zero_grad() 
                many_loss.backward(retain_graph=True)
                optimizer.step()
                # update the few head only :                
                freeze_backbone(model)
                optimizer.zero_grad()
                few_loss.backward()
                optimizer.step()
                unfreeze_backbone(model)                
                

            fewlosses.update(few_loss.item(), input.size(0))
            
                
        elif args.experts == 3:
            
            many_loss, medium_loss, few_loss = criterion(output, mask)
            loss = many_loss + medium_loss  + few_loss 
                  
            if not args.separate_backprop :
                optimizer.zero_grad() 
                loss.backward()
                optimizer.step()
            
            else :
                # update backbone with expert 1 and head for few classes with expert few
                optimizer.zero_grad() 
                many_loss.backward(retain_graph=True)
                optimizer.step()
                freeze_backbone(model)
                
                if few_loss.item() >0. :
            
                    optimizer.zero_grad()
                    few_loss.backward(retain_graph=True)
                    optimizer.step()
                    
                if medium_loss.item() >0. :
                    
                    optimizer.zero_grad()
                    medium_loss.backward()
                    optimizer.step()
                    
                unfreeze_backbone(model)                 
            
            fewlosses.update(few_loss.item(), input.size(0))            
            mediumlosses.update(medium_loss.item(), input.size(0))
        

        
        # update running metrics : 
        losses.update(loss.item(), input.size(0))
        
    # End of each epoch     
    batch_time.update(time.time() - end)    

    lr = optimizer.param_groups[0]['lr']
    lr_last = optimizer.param_groups[-1]['lr']

    msg = 'Epoch: [{0}]\t' \
        'Time {batch_time.avg:.3f}s \t' \
        'Loss {loss.avg:.5f} \t'\
        'MediumLoss {mediumloss.avg:.5f}\t'\
        'FewLoss {fewloss.avg:.5f}\t'.format(
        epoch, 
        batch_time=batch_time,
        loss=losses,
        mediumloss=mediumlosses, 
        fewloss=fewlosses)
        
    print(msg)
             
             
    if args.log_wandb :
        metrics = {
                'train_loss':losses.avg,
                'train_duration': batch_time.avg,
                'lr':lr,    
                'epoch':epoch,                         
            }
            
        wandb.log(metrics,step = epoch)
    
    
    
    



