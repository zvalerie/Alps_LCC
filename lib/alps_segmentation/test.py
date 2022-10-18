import os
import argparse
import pprint

import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from lib.core.function import test
from lib.core.loss import FocalLoss
from lib.utils.utils import create_logger

from lib.models.Unet import get_Unet
from lib.dataset.SwissImage import SwissImage


def parse_args():
    parser = argparse.ArgumentParser(description='Test image segmentation network')
    parser.add_argument('--out_dir',
                        help='directory to save outputs',
                        default='out',
                        type=str)
    parser.add_argument('--frequent',
                        help='frequency of logging',
                        default=10,
                        type=int)
    parser.add_argument('--gpus',
                        help='which gpu(s) to use',
                        default='',
                        type=str)
    parser.add_argument('--num_workers',
                        help='num of dataloader workers',
                        default=4,
                        type=int)
    args = parser.parse_args()
    
    return args

def main():
    args = parse_args()
    
    logger, tb_logger = create_logger(
        args.out_dir, phase='test', create_tf_logs=True)
    
    logger.info(pprint.pformat(args))
    
    model = get_Unet()
    
    if tb_logger:
        writer_dict = {
            'logger': tb_logger,
            'test_global_steps': 0,
            'vis_global_steps': 0,
        }
    else:
        writer_dict = None

    # Load best model
    model_state_file = os.path.join(args.out_dir,
                                    'model_best.pth.tar')
    logger.info('=> loading model from {}'.format(model_state_file))
    model.load_state_dict(torch.load(model_state_file))

    test_csv = '/data/xiaolong/master_thesis/data/test_dataset.csv'
    img_dir = '/data/xiaolong/rgb'
    dem_dir = '/data/xiaolong/dem'
    mask_dir = '/data/xiaolong/mask'
    test_dataset = SwissImage(test_csv, img_dir, dem_dir, mask_dir)

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # evaluate on validation set
    perf_indicator = test(test_loader, test_dataset, model,
                         args.out_dir, writer_dict, args)

    writer_dict['logger'].close()


if __name__ == '__main__':
    main()
