import os
import argparse
from pickle import FALSE
import pprint

import random
from xmlrpc.client import boolean
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from lib.core.function import train
from lib.core.function import validate
from lib.core.loss import FocalLoss
from lib.core.loss import CrossEntropy2D
from lib.utils.utils import create_logger
from lib.utils.utils import save_checkpoint

from lib.models.Unet import Res50_UNet
from lib.dataset.SwissImage import SwissImage

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
                        default=1e-1,
                        type=float)      
    parser.add_argument('--bs',
                        help='batch size',
                        default=64,
                        type=int)
    parser.add_argument('--out_dir',
                        help='directory to save outputs',
                        default='out',
                        type=str)
    parser.add_argument('--backbone',
                        help='backbone of encoder',
                        default='resnet50',
                        type=str)
    parser.add_argument('--frequent',
                        help='frequency of logging',
                        default=500,
                        type=int)
    parser.add_argument('--gpus',
                        help='which gpu(s) to use',
                        default='0',
                        type=str)
    # just an experience, the number of workers == cpu cores == 6 in this work station
    parser.add_argument('--num_workers',
                        help='num of dataloader workers',
                        default=6,
                        type=int)
    parser.add_argument('--continues',
                        help='continue training from checkpoint',
                        default=False,
                        type=bool)
    parser.add_argument('--debug',
                        help='is debuging?',
                        default=False,
                        type=bool)
    parser.add_argument('--tune',
                        help='is tunning?',
                        default=False,
                        type=bool)
    args = parser.parse_args()
    
    return args

def main():
    args = parse_args()
    
    logger, tb_logger = create_logger(args.out_dir, phase='train', create_tf_logs=True)
    
    logger.info(pprint.pformat(args))
    
    if args.backbone == 'resnet50':
        model = Res50_UNet(num_classes=16)
    
    if tb_logger:
        writer_dict = {
            'logger': tb_logger,
            'train_global_steps': 0,
            'valid_global_steps': 0,
            'vis_global_steps': 0,
        }
    else:
        writer_dict = None
    
    if len(args.gpus) == 0:
        gpus = []
    else:
        gpus = [int(i) for i in args.gpus.split(',')]       
    
    # Define loss function (criterion) and optimizer  
    if len(gpus) > 0:
        model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
        criterion = CrossEntropy2D().cuda()
    else:
        criterion = CrossEntropy2D()
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    
    # Create training and validation datasets
    if args.tune:
        train_csv = '/data/xiaolong/master_thesis/data/subset/train_subset.csv'
        val_csv = '/data/xiaolong/master_thesis/data/subset/val_subset.csv'
    else : 
        train_csv = '/data/xiaolong/master_thesis/data/train_dataset.csv'
        val_csv = '/data/xiaolong/master_thesis/data/val_dataset.csv'
        
    img_dir = '/data/xiaolong/rgb'
    dem_dir = '/data/xiaolong/dem'
    mask_dir = '/data/xiaolong/mask'
    train_dataset = SwissImage(train_csv, img_dir, dem_dir, mask_dir, args.debug)
    val_dataset = SwissImage(val_csv, img_dir, dem_dir, mask_dir, args.debug)
    
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
        num_workers=1,
        pin_memory=True
    )
    
    best_perf = 0.0 # best performance so far (mean IoU)
    best_model = False
    start_epoch = 0
    
    #load existing model
    if args.continues:
        log_dir = '/data/xiaolong/master_thesis/out/checkpoint.pth.tar'
        if os.path.exists(log_dir):
            checkpoint = torch.load(log_dir)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            perf_indicator = checkpoint['perf']
            print('loadding epoch {} succeed'.format(start_epoch))
        else:
            print('loadding checkpoint failed, start trainning from epoch 0')
            start_epoch = 0
    else:
        start_epoch = 0

    for epoch in range(start_epoch, args.epoch):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch,
              args.out_dir, writer_dict, args)

        if (epoch + 1) % 1 == 0:
            # evaluate on validation set
            perf_indicator = validate(val_loader, val_dataset, model,
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

        # update best model so far
        logger.info('=> saving checkpoint to {}'.format(args.out_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'perf': perf_indicator,
            'last_epoch': epoch,
            'optimizer': optimizer.state_dict(),
        }, best_model, args.out_dir)


    final_model_state_file = os.path.join(args.out_dir,
                                          'final_state.pth.tar')
    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    torch.save(model.module.state_dict() if len(gpus) > 0 else model.state_dict(), final_model_state_file)
    writer_dict['logger'].close()


if __name__ == '__main__':
    main()