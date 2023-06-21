"""
    My implementation of training pipeline
"""
import os

import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler
from torch.optim.lr_scheduler import StepLR, LambdaLR
from pprint import pprint
from copy import deepcopy

from utils.training_utils import get_dataloader, get_model, get_criterion, set_all_random_seeds, setup_wandb_log
from utils.argparser import parse_args

from XL.lib.core.loss import ResCELoss, ResCELoss_3exp

from utils.train_fn import train_ACE
from utils.validate_fn import validate_ACE
from utils.test_fn import test_ACE



def main(args):
   
    # set all random seeds :
    set_all_random_seeds(args.seed)
    setup_wandb_log(args)
    
    
    # Choose model and device :
    model = get_model(args)
    device =  torch.device("cuda") if torch.cuda.is_available() or not args.force_cpu else 'cpu'
    model = model.to(device)
    
    # Get dataloaders : 
    train_loader = get_dataloader(args,phase='train')
    val_loader = get_dataloader(args=args,phase='val')
    
       
    # # Define loss function (criterion) and optimizer
    criterion = get_criterion (args)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.lr_decay_rate)
       
    if args.experts == 3: 
        many_index = [1, 5, 7, 8, 9]
        medium_index = [2, 6]
        few_index = [3, 4]
        ls_index = [many_index, medium_index, few_index]
        criterion = ResCELoss_3exp(many_index, medium_index, few_index, args=args).to(device)
        lr_ratio = [0.03, 0.01] ## ratio of rare categories to frequent categories
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    
    
    # Start model training :     
    best_miou = 0.0 # best performance so far (mean IoU)
    start_epoch = 0
    miou=0.

    for epoch in range( args.epoch):
        
        # train for one epoch
        train_ACE(train_loader, model, criterion, optimizer, epoch, args, device)
        scheduler.step()
        
        # evaluate on validation set
        val_loss, miou = validate_ACE(val_loader, model, criterion, epoch, args, device)

        # update best model if best performances : 
        miou=0
        if miou > best_miou :
                best_miou = miou
                model_checkpoint = {
                        'epoch': epoch ,
                        'state_dict': deepcopy(model.state_dict()),
                        'perf': miou,
                        'last_epoch': epoch,
                        'optimizer': optimizer.state_dict(),
                        }
                torch.save(model_checkpoint, os.path.join( args.out_dir, args.name,'ep_{}'.format(epoch)+'.pt'))

        


    # End of training : save final model
    model_checkpoint = {
                        'epoch': epoch ,
                        'state_dict': deepcopy (model.state_dict()),
                        'perf': miou,
                        'last_epoch': epoch,
                        'optimizer': optimizer.state_dict(),
                        }
    torch.save(model_checkpoint, os.path.join( args.out_dir, args.name,'final.pt'))
    print('End of model training')
    
    test_ACE(model,args,device)


if __name__ == '__main__':
    args = parse_args()
    pprint(vars ( args))
    main(args)
        