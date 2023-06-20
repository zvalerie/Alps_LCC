import os
import argparse
import pprint

import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from lib.utils.utils import create_logger
from lib.core.function_ACE import test

from lib.models.ACE_UNet import ACE_Res50_UNet
from lib.models.ACE_DeepLabv3P import ACE_deeplabv3P_resnet
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
                        default='out/ACE',
                        type=str)
    parser.add_argument('--model_path',
                        help='time_str when trainning model ',
                        default='',
                        type=str)
    parser.add_argument('--bs',
                        help='batch size',
                        default=16,
                        type=int)
    parser.add_argument('--model',
                        help='backbone of encoder',
                        default='Unet',
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
    args = parser.parse_args()
    
    return args

def main():
    args = parse_args()
    # create logger
    logger, tb_logger, time_str = create_logger(
        args.out_dir, phase='test', create_tf_logs=True)
    
    logger.info(pprint.pformat(args))
    logger.info('Test ACE model')
    if tb_logger:
        writer_dict = {
            'logger': tb_logger,
            'test_global_steps': 0,
            'vis_global_steps': 0,
        }
    else:
        writer_dict = None
    
    # define model
    if args.model == 'Unet':
        model = ACE_Res50_UNet(num_classes=10, num_experts=args.experts, train_MLP=False, pretrained = True)    
    elif args.model == 'Deeplabv3':
        model = ACE_deeplabv3P_resnet(num_classes=10, output_stride=8, pretrained_backbone=True, num_experts=args.experts, is_MLP=False)
    else:
        raise NotImplementedError('Model not supported: {}'.format(args.model))
    
    device = torch.device("cuda")
    model = model.to(device)
    
    # define target classes for each expert
    if args.experts==2:
        many_index = [0, 1, 5, 8, 9]
        few_index = [2, 3, 4, 6, 7]
        ls_index = [many_index, few_index]
    elif args.experts==3:
        many_index = [0, 1, 5, 8, 9]
        medium_index = [2, 6, 7]
        few_index = [3, 4]
        ls_index = [many_index, medium_index, few_index]
    else:
        raise ValueError('Number of experts not supported: {}'.format(args.experts))
    

    # load best model
    model_state_file = os.path.join(args.out_dir, args.model_path,
                                    'model_best.pth.tar')
    logger.info('=> loading model from {}'.format(model_state_file))
    bestmodel = torch.load(model_state_file)
    model.load_state_dict(bestmodel, strict=False)

    # data path
    if args.tune:
        test_csv = '/data/xiaolong/master_thesis/data_preprocessing/subset/test_subset.csv'
    else :
        test_csv = '/data/xiaolong/master_thesis/data_preprocessing/test_dataset.csv'
    img_dir = '/data/xiaolong/rgb'
    dem_dir = '/data/xiaolong/dem'
    mask_dir = '/data/xiaolong/mask'
    
    # define dataset and dataloader
    test_dataset = SwissImage(test_csv, img_dir, dem_dir, mask_dir, debug=args.debug)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # evaluate on test set
    confusionMatrix = test(test_loader, test_dataset, model, ls_index,
                         args.out_dir, writer_dict, args)

    
    # np.save('/data/xiaolong/master_thesis/confusion_matrix/' + 'ACE_cm' + time_str, confusionMatrix)
    writer_dict['logger'].close()

if __name__ == '__main__':
    main()
