"""
    My implementation of training pipeline
    
    
    Look at the official one : 
    https://github.com/winterxx/ACE/blob/main/main/train.py
    Xiaolong one : 
    https://github.com/xiaollu/Alps_LCC
    
    
"""
import os
import torch
from torch.optim.lr_scheduler import StepLR, LambdaLR, ReduceLROnPlateau
from pprint import pprint
from copy import deepcopy

from utils.training_utils import get_dataloader, get_model, get_criterion, set_all_random_seeds, setup_wandb_log, get_optimizer 
from utils.argparser import parse_args


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
    optimizer = get_optimizer(model=model,args=args)
    scheduler = ReduceLROnPlateau(optimizer, min_lr = 1e-8, factor = args.lr_decay_rate, threshold = 1e-2, verbose = True)
    
    ## Show experiment setup : 
    print('*'*80)    
    print('Experiment set up : ') 
    print('\tModel    :  ', type(model).__name__,)
    print('\tNb Expert:  ', args.experts)
    print('\tLoss     :  ', type(criterion).__name__) 
    print('\tOptimizer:  ', type(optimizer).__name__) 
    print('\tDevice   :  ', device)      
    print('*'*80)
    
    
    # Start model training :     
    best_miou = 0.0 # best performance so far (mean IoU)
    start_epoch = 0
    miou=0.
    
    if not args.test_only: 
        for epoch in range( args.epoch):
                   
            # train for one epoch
            train_ACE(train_loader, model, criterion, optimizer, epoch, args)
            
            
            # evaluate on validation set
            val_loss, miou = validate_ACE(val_loader, model, criterion, epoch, args)
            
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
                print('Model saved, best miou')
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
    
    test_ACE(model,args)


if __name__ == '__main__':
    args = parse_args()

    main(args)
        