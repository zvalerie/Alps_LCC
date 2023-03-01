import os
import argparse
import pprint

import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR

from lib.core.function import train
from lib.core.function import validate
# from lib.core.loss import FocalLoss
from lib.core.loss import CrossEntropy2D, ResCELoss, ResCELoss_3exp, SeesawLoss
from lib.utils.utils import get_optimizer
from lib.utils.utils import create_logger
from lib.utils.utils import save_checkpoint

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
    parser = argparse.ArgumentParser(description='Re-train the segmentation head')
    parser.add_argument('--lr',
                        help='learning rate',
                        default=1e-5,
                        type=float)
    parser.add_argument('--epoch',
                        help='training epoches',
                        default=20,
                        type=int)  
    parser.add_argument('--wd',
                        help='weight decay',
                        default=1e-2,
                        type=float)      
    parser.add_argument('--bs',
                        help='batch size',
                        default=32,
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
                        default='out/LWS',
                        type=str)
    parser.add_argument('--model',
                        help='model to be used',
                        default='Unet',
                        type=str)
    parser.add_argument('--frequent',
                        help='frequency of logging',
                        default=100,
                        type=int)
    parser.add_argument('--num_workers',
                        help='num of dataloader workers',
                        default=6,
                        type=int)
    parser.add_argument('--debug',
                        help='is debuging?',
                        default=False,
                        type=bool)
    parser.add_argument('--tune',
                        help='is tunning?',
                        default=True,
                        type=bool)
    parser.add_argument('--loss',
                        help='complementary loss type',
                        default=1,
                        type=int)
    parser.add_argument('--experts',
                        help='number of experts',
                        default=2,
                        type=int)
    parser.add_argument('--comFactor',
                        help='factor of comLoss',
                        default=0,
                        type=int)
    parser.add_argument('--TRAIN_SEP',
                        help='train experts seperately',
                        default=False,
                        type=bool)
    parser.add_argument('--MLP',
                        help='Train MLP layer to combine the results of experts',
                        default=False,
                        type=bool)
    parser.add_argument('--LWS',
                        help='Train LWS layer to retune the classifier',
                        default=False,
                        type=bool)
    parser.add_argument('--model_name',
                        help='model name for retuning',
                        default=None,
                        type=str)
    args = parser.parse_args()
    
    return args

def main():
    args = parse_args()

    logger, tb_logger, folder_name = create_logger(args.out_dir, phase='train', create_tf_logs=True)
    logger.info(pprint.pformat(args))
    logger.info('MLP, experts:{}'.format(args.experts))
    if args.model == 'Unet':
        model = ACE_Res50_UNet(num_classes=10, train_LWS = args.LWS, train_MLP = args.MLP, num_experts=args.experts, pretrained = False)
    elif args.model =='Deeplabv3':
        model = ACE_deeplabv3P_resnet(num_classes=10, output_stride=8, pretrained_backbone=False, num_experts=args.experts, is_MLP=True)
    model_path = '/data/xiaolong/master_thesis/out/ACE/' + args.model_name + '/model_best.pth.tar'
    pretrained_dict = torch.load(model_path)
    
    # if args.LWS==True:
    #     for k in list(pretrained_dict.keys()):
    #         if k.startswith("SegHead"):
    #             k_split = k.split('.')
    #             new_k = k_split[0] + '.conv2d.' + k_split[1]
    #             pretrained_dict[new_k] = pretrained_dict.pop(k)
        
    #     # only train the scales in LWS (by setting requires_grad of all the other layers to False)
    #     model.load_state_dict(pretrained_dict, strict=False)
    #     for k, v in model.named_parameters():
    #         if not k.startswith("SegHead"):
    #             v.requires_grad=False
                
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name, param)
    
    if args.MLP==True:
        pretrained_dict.pop('classifier.MLP.mlp.0.weight')
        pretrained_dict.pop('classifier.MLP.mlp.0.bias')
        pretrained_dict.pop('classifier.MLP.mlp.2.weight')
        pretrained_dict.pop('classifier.MLP.mlp.2.bias')
        model.load_state_dict(pretrained_dict, strict=False)
        for k, v in model.named_parameters():
            if not k.startswith("classifier.MLP"):
                v.requires_grad=False
            # if k.startswith("classifier"):
            #     print(k)   
        
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name)            
                    
    writer_dict = {
            'logger': tb_logger,
            'train_global_steps': 0,
            'valid_global_steps': 0,
            'vis_global_steps': 0,
            }   
    
    # Define loss function (criterion) and optimizer  
    device = torch.device("cuda")
    model = model.to(device)
    
    if args.MLP == True:
        # criterion_exp = CrossEntropy2D(ignore_index=-1).to(device)
        criterion = CrossEntropy2D(ignore_index=-1).to(device)
        # criterion = SeesawLoss().to(device)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd)    
    # elif args.experts == 2: 
    #     many_index = [1, 5, 8, 9]
    #     few_index = [2, 3, 4, 6, 7]
    #     ls_index = [many_index, few_index]
    #     criterion = ResCELoss(many_index, few_index, args=args).to(device)
    #     lr_ratio = [0.03] ## ratio of rare categories to frequent categories
    #     optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd)
    #     # optimizer = get_optimizer(model, "ADAM", num_experts=args.experts, base_lr=args.lr, lr_ratio=lr_ratio)
    # elif args.experts == 3: 
    #     many_index = [1, 5, 8, 9]
    #     medium_index = [6, 7]
    #     few_index = [2, 3, 4]
    #     ls_index = [many_index, medium_index, few_index]
    #     criterion = ResCELoss_3exp(many_index, medium_index, few_index, args=args).to(device)
    #     lr_ratio = [0.03, 0.01] ## ratio of rare categories to frequent categories
    #     optimizer = get_optimizer(model, "ADAM", num_experts=args.experts, base_lr=args.lr, lr_ratio=lr_ratio)
    #     # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.lr_decay_rate)
    
    # Create training and validation datasets
    if args.tune:
        train_csv = '/data/xiaolong/master_thesis/data_preprocessing/subset/train_subset.csv'
        val_csv = '/data/xiaolong/master_thesis/data_preprocessing/subset/val_subset.csv'
    else : 
        train_csv = '/data/xiaolong/master_thesis/data_preprocessing/train_dataset.csv'
        val_csv = '/data/xiaolong/master_thesis/data_preprocessing/val_dataset.csv'
        
    img_dir = '/data/xiaolong/rgb'
    dem_dir = '/data/xiaolong/dem'
    mask_dir = '/data/xiaolong/mask'
    
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
    
    best_perf = 0.0 # best performance so far (mean IoU)
    best_model = False
    start_epoch = 0

    for epoch in range(start_epoch, args.epoch):
        # train for one epoch
        train(train_loader,train_dataset, model, criterion, optimizer, epoch,
              args.out_dir, writer_dict, scheduler, args=args)

        if (epoch + 1) % 1 == 0:
            # evaluate on validation set
            val_loss, perf_indicator = validate(val_loader, val_dataset, model,
                                      criterion, args.out_dir, writer_dict, args)

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
        }, best_model, folder_path)

    final_model_state_file = os.path.join(folder_path,
                                          'final_state.pth.tar')
    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    torch.save(model.state_dict(), final_model_state_file)
    writer_dict['logger'].close()

if __name__ == '__main__':
    main()