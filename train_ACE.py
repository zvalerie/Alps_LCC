import os
import argparse
import pprint

import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR

from lib.core.function_ACE import train, validate
# from lib.core.loss import FocalLoss
from lib.core.loss import ResCELoss, ResCELoss_3exp
from lib.utils.utils import create_logger, save_checkpoint, weighted_sampler

from lib.models.ACE_UNet import ACE_Res50_UNet
from lib.models.ACE_DeepLabv3P import ACE_deeplabv3P_resnet
from lib.dataset.SwissImage import SwissImage
from lib.utils.transforms import Compose, MyRandomRotation90, MyRandomHorizontalFlip, MyRandomVerticalFlip


# fix random seeds for reproducibility
seed = 1
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser(description='Train image segmentation network')
    parser.add_argument('--lr',
                        help='learning rate',
                        default=1e-5,
                        type=float)
    parser.add_argument('--epoch',
                        help='training epoches',
                        default=50,
                        type=int)  
    parser.add_argument('--wd',
                        help='weight decay',
                        default=1e-2,
                        type=float)      
    parser.add_argument('--bs',
                        help='batch size',
                        default=16,
                        type=int)
    parser.add_argument('--lr_decay_rate',
                        help='scheduler_decay_rate',
                        default=0.1,
                        type=float)
    parser.add_argument('--step_size',
                        help='step to decrease lr',
                        default = 10,
                        type=int)
    parser.add_argument('--out_dir',
                        help='directory to save outputs',
                        default='out/ACE',
                        type=str)
    parser.add_argument('--model',
                        help='model name',
                        default='Unet',
                        type=str)
    parser.add_argument('--train_sep',
                            help='train experts separately',
                            default=False,
                            type=bool)
    parser.add_argument('--frequent',
                        help='frequency of logging',
                        default=100,
                        type=int)
    parser.add_argument('--num_workers',
                        help='num of dataloader workers',
                        default=6,
                        type=int)
    parser.add_argument('--debug',
                        help='debug mode',
                        default=False,
                        type=bool)
    parser.add_argument('--tune',
                        help='tune mode',
                        default=True,
                        type=bool)
    parser.add_argument('--experts',
                        help='number of experts',
                        default=2,
                        type=int)
    parser.add_argument('--is_weighted_sampler',
                        help='is_weighted_sampler',
                        default=False,
                        type=bool)
    parser.add_argument('--fewloss_factor',
                        help='factor to adjust the weight of few loss',
                        default=1,
                        type=float)
    parser.add_argument('--mediumloss_factor',
                        help='factor to adjust the weight of few loss',
                        default=1,
                        type=float)
    args = parser.parse_args()
    
    return args

def main():
    args = parse_args()

    # Create logger
    logger, tb_logger, folder_name = create_logger(args.out_dir, phase='train', create_tf_logs=True)
    logger.info(pprint.pformat(args))

    writer_dict = {
        'logger': tb_logger,
        'train_global_steps': 0,
        'valid_global_steps': 0,
        'vis_global_steps': 0,
        }  

    # Define model
    if args.model == 'Unet':
        model = ACE_Res50_UNet(num_classes=10, num_experts=args.experts, is_MLP = False, pretrained = True)
    elif args.model == 'Deeplabv3':
        model = ACE_deeplabv3P_resnet(num_classes=10, output_stride=8, pretrained_backbone=True, num_experts=args.experts, is_MLP=False)
 
    device = torch.device("cuda")
    model = model.to(device)
    
    # Define target classes for each expert, loss function and optimizer
    if args.experts == 2: 
        many_index = [1, 5, 8, 9]
        few_index = [2, 3, 4, 6, 7]
        ls_index = [many_index, few_index]
        criterion = ResCELoss(many_index, few_index, args=args).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.experts == 3: 
        many_index = [1, 5, 8, 9]
        medium_index = [2, 6, 7]
        few_index = [3, 4]
        ls_index = [many_index, medium_index, few_index]
        criterion = ResCELoss_3exp(many_index, medium_index, few_index, args=args).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    else: 
        raise ValueError('Number of experts should be 2 or 3')
    
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.lr_decay_rate)    
    
    # Data path
    if args.tune:
        train_csv = '/data/xiaolong/master_thesis/data_preprocessing/subset/train_subset_few.csv'
        val_csv = '/data/xiaolong/master_thesis/data_preprocessing/subset/val_subset.csv'
    else : 
        train_csv = '/data/xiaolong/master_thesis/data_preprocessing/train_dataset.csv'
        val_csv = '/data/xiaolong/master_thesis/data_preprocessing/val_dataset.csv'
        
    img_dir = '/data/xiaolong/rgb'
    dem_dir = '/data/xiaolong/dem'
    mask_dir = '/data/xiaolong/mask'
    
    # Data augmentation
    common_transform = Compose([
        MyRandomHorizontalFlip(p=0.5),
        MyRandomVerticalFlip(p=0.5),
        MyRandomRotation90(p=0.5),
        ])
        
    img_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)
        ])

    # Define dataset and dataloader
    train_dataset = SwissImage(train_csv, img_dir, dem_dir, mask_dir, common_transform=common_transform, img_transform=img_transform, debug=args.debug)
    val_dataset = SwissImage(val_csv, img_dir, dem_dir, mask_dir, debug=args.debug)
    
    if args.is_weighted_sampler:
        weight = weighted_sampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.bs,
            sampler=WeightedRandomSampler(weight, len(weight)),
            num_workers=args.num_workers,
            pin_memory=True
        )
    else: 
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
    
    best_perf = 0.0 # best performance so far (mean IoU)
    best_model = False
    start_epoch = 0

    for epoch in range(start_epoch, args.epoch):
        # train for one epoch
        train(train_loader,train_dataset, model, criterion, optimizer, epoch,
              args.out_dir, writer_dict, args)

        if (epoch + 1) % 1 == 0:
            # evaluate on validation set
            val_loss, perf_indicator = validate(val_loader, val_dataset, model,
                                      criterion, ls_index, args.out_dir, writer_dict, args)

            # update best performance
            if perf_indicator > best_perf:
                best_perf = perf_indicator
                best_model = True
            else:
                best_model = False
        else:
            perf_indicator = -1
            best_model = False
        
        scheduler.step()
        
        # update best model so far
        folder_path = os.path.join(args.out_dir, folder_name)
        logger.info('=> saving checkpoint to {}'.format(folder_path))
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'perf': perf_indicator,
            'last_epoch': epoch,
            'optimizer': optimizer.state_dict(),
            # 'many_optimizer': optimizer[0].state_dict(),
            # 'few_optimizer': optimizer[1].state_dict(),
        }, best_model, folder_path)

    final_model_state_file = os.path.join(folder_path,
                                          'final_state.pth.tar')
    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    torch.save(model.state_dict(), final_model_state_file)
    writer_dict['logger'].close()

if __name__ == '__main__':
    main()