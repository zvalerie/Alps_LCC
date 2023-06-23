"""
    My implementation of training pipeline
"""
import os
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler
from torch.optim.lr_scheduler import StepLR, LambdaLR, ReduceLROnPlateau
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
    pprint(vars(args))
    
    
    # Choose model and device :
    model = get_model(args)
    device =  torch.device("cuda") if torch.cuda.is_available() or not args.force_cpu else 'cpu'
    args.device = device
    model = model.to(device)
    
    # Get dataloaders : 
    train_loader = get_dataloader(args,phase='train')
    val_loader   = get_dataloader(args,phase='val')
    
       
    # # Define loss function (criterion) and optimizer
    criterion = get_criterion (args)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
   # scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.lr_decay_rate)
    scheduler = ReduceLROnPlateau(optimizer, min_lr = 1e-6, factor = args.lr_decay_rate, threshold = 1e-2, verbose = True)
       
    if args.experts == 3: 

        lr_ratio = [0.03, 0.01] ## ratio of rare categories to frequent categories
        
    
    
    # Start model training :     
    best_miou = 0.0 # best performance so far (mean IoU)
    start_epoch = 0
    miou=0.
    if not args.test_only: 
       
        for epoch in range( args.epoch):
                   
            # train for one epoch
            train_ACE(train_loader, model, criterion, optimizer, epoch, args, device)
            
            
            # evaluate on validation set
            val_loss, miou = validate_ACE(val_loader, model, criterion, epoch, args, device)
            
            scheduler.step(metrics=val_loss)
            
            # update best model if best performances : 
            model_checkpoint = {
                            'epoch': epoch ,
                            'state_dict': deepcopy(model.state_dict()),
                            'perf': miou,
                            'last_epoch': epoch,
                            'optimizer': optimizer.state_dict(),
                            }
            torch.save(model_checkpoint, os.path.join( args.out_dir, args.name,'last_model.pt'))
            if miou > best_miou and epoch>10 :
                best_miou = miou
                torch.save(model_checkpoint, os.path.join( args.out_dir, args.name,'current_best.pt'))

        
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
    args
    args.log_wandb = False
    args.debug=True
    args.test_only = True
    main(args)
        