"""
    My implementation of training pipeline
"""
import os
import argparse
import pprint

import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR, LambdaLR

from XL.lib.core.function import train
from XL.lib.core.function import validate
# from XL.lib.core.loss import FocalLoss
from XL.lib.core.loss import CrossEntropy2D, SeesawLoss, PixelPrototypeCELoss
from XL.lib.utils.utils import create_logger
from XL.lib.utils.utils import save_checkpoint

from XL.lib.models.Unet import Res50_UNet
from XL.lib.models.DeepLabv3Plus import deeplabv3P_resnet
#from XL.lib.models.DeepLabv3Proto import deeplabv3P_resnet_proto
from XL.lib.dataset.SwissImage import SwissImage
from XL.lib.utils.transforms import Compose, MyRandomRotation90, MyRandomHorizontalFlip, MyRandomVerticalFlip
from XL.lib.core.loss import ResCELoss, ResCELoss_3exp
from utils.argparser import parse_args

# fix random seeds for reproducibility
seed = 1
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)



def main(args):
   

    # Choose model : 
    if args.model == 'Unet':
        model = Res50_UNet(num_classes=10)
    elif args.model == 'Deeplabv3':
        model = deeplabv3P_resnet(num_classes=10, output_stride=8, pretrained_backbone=True)
    elif args.model == 'Deeplabv3_proto':
       raise NotImplementedError
       # model = deeplabv3P_resnet_proto(num_classes=10, output_stride=8, pretrained_backbone=True)


   # choose Device :
    device =  torch.device("cuda") if torch.cuda.is_available() or not args.force_cpu else 'cpu'
    model = model.to(device)
    
    
    
    # Define loss function (criterion) and optimizer
    if args.loss == 'celoss':
        criterion = CrossEntropy2D(ignore_index=0).to(device)
    elif args.loss == 'seesawloss':
        criterion = SeesawLoss(ignore_index=0).to(device)
    elif args.model =='Deeplabv3_proto':
        criterion = PixelPrototypeCELoss().to(device)
        
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.lr_decay_rate)

    if args.experts == 3: 
        many_index = [1, 5, 7, 8, 9]
        medium_index = [2, 6]
        few_index = [3, 4]
        ls_index = [many_index, medium_index, few_index]
        criterion = ResCELoss_3exp(many_index, medium_index, few_index, args=args).to(device)
        lr_ratio = [0.03, 0.01] ## ratio of rare categories to frequent categories



    # Create training and validation datasets       
    img_dir = '/home/valerie/data/rocky_tlm/rgb/'  #'/data/xiaolong/rgb'
    dem_dir = '/home/valerie/data/rocky_tlm/dem/' # /data/xiaolong/dem'
    mask_dir = '/home/valerie/data/ace_alps/mask'
    
    if args.tune:
        train_csv = 'data/split_subset/train_subset.csv'
        val_csv = '/home/valerie/Projects/Alps_LCC/data/split_subset/val_subset.csv'
    else : 
        train_csv = 'data/split/train_dataset.csv'
        val_csv = 'data/split/val_dataset.csv'  
    
    common_transform = Compose([
        MyRandomHorizontalFlip(p=0.5),
        MyRandomVerticalFlip(p=0.5),
        MyRandomRotation90(p=0.5),
        ])
        
    img_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)
        ]) 
    train_dataset = SwissImage(train_csv, img_dir, dem_dir, mask_dir, common_transform=common_transform, img_transform=img_transform, debug=args.debug)
    val_dataset = SwissImage(val_csv, img_dir, dem_dir, mask_dir, debug=args.debug)
    train_loader = DataLoader(
            train_dataset,
            batch_size=args.bs,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    
    # Start model training : 
    
    best_perf = 0.0 # best performance so far (mean IoU)
    best_model = False
    start_epoch = 0
    writer_dict =None
    folder_name =''
    
    for epoch in range(start_epoch, args.epoch):
        
        # train for one epoch
        train(train_loader,train_dataset, model, criterion, optimizer, epoch,
              args.out_dir, writer_dict, scheduler, args)

        
        # evaluate on validation set
        val_loss, perf_indicator = validate(val_loader, val_dataset, model,
                                      criterion, args.out_dir, writer_dict, args)

        # update best performance
        if perf_indicator > best_perf:
                best_perf = perf_indicator
                best_model = True
        else:
                best_model = False

        
        scheduler.step()
        
        # update best model so farr
        folder_path = os.path.join(args.out_dir, folder_name)
        
        save_checkpoint(
            {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'perf': perf_indicator,
            'last_epoch': epoch,
            'optimizer': optimizer.state_dict(),
            }, 
            best_model, 
            folder_path)

    # End of training : save final model
    final_model_state_file = os.path.join(folder_path, 'final_state.pth.tar')

    torch.save(model.state_dict(), final_model_state_file)


if __name__ == '__main__':
    args = parse_args()
    main(args)
        