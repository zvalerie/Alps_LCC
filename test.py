import os
import argparse
import pprint

import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from lib.utils.utils import create_logger
from lib.core.function import test

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
    parser.add_argument('--out_dir',
                        help='directory to save outputs',
                        default='out/test',
                        type=str)
    parser.add_argument('--foldername',
                        help='time_str when trainning model ',
                        default='',
                        type=str)
    parser.add_argument('--bs',
                        help='batch size',
                        default=32,
                        type=int)
    parser.add_argument('--backbone',
                        help='backbone of encoder',
                        default='resnet50',
                        type=str)
    parser.add_argument('--frequent',
                        help='frequency of logging',
                        default=100,
                        type=int)
    # just an experience, the number of workers == cpu cores == 6 in this work station
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
                        default=False,
                        type=bool)
    args = parser.parse_args()
    
    return args

def main():
    args = parse_args()
    
    logger, tb_logger, _ = create_logger(
        args.out_dir, phase='test', create_tf_logs=True)
    
    logger.info(pprint.pformat(args))
    
    if args.backbone == 'resnet50':
        model = Res50_UNet(num_classes=10)
        
    # Define loss function (criterion) and optimizer  
    device = ("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
        
    if tb_logger:
        writer_dict = {
            'logger': tb_logger,
            'test_global_steps': 0,
            'vis_global_steps': 0,
        }
    else:
        writer_dict = None

    # Load best model
    model_state_file = os.path.join(args.out_dir, args.foldername,
                                    'model_best.pth.tar')
    logger.info('=> loading model from {}'.format(model_state_file))
    bestmodel = torch.load(model_state_file)
    model.load_state_dict(bestmodel)

    if args.tune:
        test_csv = '/data/xiaolong/master_thesis/data_preprocessing/subset/test_subset.csv'
    else :
        test_csv = '/data/xiaolong/master_thesis/data_preprocessing/test_dataset.csv'
    img_dir = '/data/xiaolong/rgb'
    dem_dir = '/data/xiaolong/dem'
    mask_dir = '/data/xiaolong/mask'
    test_dataset = SwissImage(test_csv, img_dir, dem_dir, mask_dir, debug=args.debug)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # evaluate on test set
    confusionMatrix = test(test_loader, test_dataset, model,
                         args.out_dir, writer_dict, args)

    
    np.save('ConfusionMatrix', confusionMatrix)
    writer_dict['logger'].close()

if __name__ == '__main__':
    main()
