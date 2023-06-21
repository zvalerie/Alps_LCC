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

from utils.training_utils import get_dataloader, get_model, get_criterion, set_all_random_seeds
from utils.argparser import parse_args

#from XL.lib.core.function import train
from XL.lib.core.function import validate
# from XL.lib.core.loss import FocalLoss
from utils.training_utils import save_checkpoint
from XL.lib.core.loss import ResCELoss, ResCELoss_3exp

from utils.train_fn import train_ACE, validate_ACE



def main(args):
   
    # set all random seeds :
    set_all_random_seeds(args.seed)
    
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
    writer_dict =None

    for epoch in range( args.epoch):
        
        # train for one epoch
        train_ACE(train_loader, model, criterion, optimizer, epoch, args, device)

        
        # evaluate on validation set
        val_loss, miou = validate_ACE(val_loader, model, criterion, epoch, args, device)

        # update best performance
        if miou > best_miou:
                best_miou = miou
                model_checkpoint = {
                        'epoch': epoch ,
                        'state_dict': model.state_dict().deepcopy(),
                        'perf': miou,
                        'last_epoch': epoch,
                        'optimizer': optimizer.state_dict(),
                        }
                torch.save(model_checkpoint, os.path.join( args.outdir, args.name,'ep_'{epoch},'.pt'))

        scheduler.step()


    # End of training : save final model
    model_checkpoint = {
                        'epoch': epoch ,
                        'state_dict': model.state_dict().deepcopy(),
                        'perf': miou,
                        'last_epoch': epoch,
                        'optimizer': optimizer.state_dict(),
                        }
    torch.save(model_checkpoint, os.path.join( args.outdir, args.name,'_final.pt'))
    print('end of model training')


if __name__ == '__main__':
    args = parse_args()
    main(args)
        