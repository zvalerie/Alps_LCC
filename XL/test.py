import os
import argparse
import pprint

import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from lib.utils.utils import create_logger
from lib.core.function import test, ratio_acc_test

from lib.models.Unet import Res50_UNet
from lib.models.ACE_UNet import ACE_Res50_UNet
from lib.models.ACE_DeepLabv3P import ACE_deeplabv3P_resnet
from lib.models.DeepLabv3Plus import deeplabv3P_resnet
from lib.models.DeepLabv3Proto import deeplabv3P_resnet_proto
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
                        default='out/train',
                        type=str)
    parser.add_argument('--model_name',
                        help='time_str when trainning model ',
                        default='',
                        type=str)
    parser.add_argument('--bs',
                        help='batch size',
                        default=16,
                        type=int)
    parser.add_argument('--model',
                        help='model to be trained',
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
                        help='is debuging?',
                        default=False,
                        type=bool)
    parser.add_argument('--tune',
                        help='is tunning?',
                        default=True,
                        type=bool)
    parser.add_argument('--experts',
                        help='number of experts?',
                        default=2,
                        type=int)
    parser.add_argument('--MLP',
                        help='Train MLP layer to combine the results of experts',
                        default=False,
                        type=bool)
    args = parser.parse_args()
    
    return args

def main():
    args = parse_args()
    
    logger, tb_logger, time_str = create_logger(
        args.out_dir, phase='test', create_tf_logs=True)
    
    logger.info(pprint.pformat(args))
    
    if args.model == 'Unet':
        model = Res50_UNet(num_classes=10)
        if args.MLP == True:
            model = ACE_Res50_UNet(num_classes=10, train_LWS = False, train_MLP = args.MLP, num_experts=args.experts, pretrained = False)
    elif args.model == 'Deeplabv3':
        model = deeplabv3P_resnet(num_classes=10, output_stride=8, pretrained_backbone=True)
        if args.MLP == True:
            model = ACE_deeplabv3P_resnet(num_classes=10, output_stride=8, pretrained_backbone=True, num_experts=args.experts)
    elif args.model == 'Deeplabv3_proto':
        model = deeplabv3P_resnet_proto(num_classes=10, output_stride=8, pretrained_backbone=True)
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
    model_state_file = os.path.join(args.out_dir, args.model_name,
                                    'model_best.pth.tar')
    logger.info('=> loading model from {}'.format(model_state_file))
    bestmodel = torch.load(model_state_file)
    model.load_state_dict(bestmodel)

    if args.tune:
        test_csv = '/data/xiaolong/master_thesis/data_preprocessing/subset/test_subset.csv'
        # test_csv = '/data/xiaolong/master_thesis/data_preprocessing/4_test_dataset.csv'
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
    # ratio_acc_test(test_loader, test_dataset, model,
    #                      args.out_dir, writer_dict, args)
    
    # np.save('/data/xiaolong/master_thesis/confusion_matrix/' + time_str, confusionMatrix)
    writer_dict['logger'].close()

if __name__ == '__main__':
    main()
