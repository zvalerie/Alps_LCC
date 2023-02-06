import os
import argparse
import pprint
import logging
import time

import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from numpy import linalg as LA

from lib.utils.utils import create_logger
from lib.core.function_ACE import test

from lib.models.ACE_UNet import ACE_Res50_UNet
from lib.models.ACE_DeepLabv3P import deeplabv3P_resnet
from lib.core.inference import get_final_preds
from lib.utils.vis import vis_seg_mask
from lib.dataset.SwissImage import SwissImage
from lib.utils.evaluation import MetricLogger
from lib.core.function_ACE import AverageMeter


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
    parser.add_argument('--foldername',
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
                        help='is debuging?',
                        default=False,
                        type=bool)
    parser.add_argument('--tune',
                        help='is tunning?',
                        default=True,
                        type=bool)
    parser.add_argument('--experts',
                        help='number of experts',
                        default=2,
                        type=int)
    parser.add_argument('--LWS',
                        help='train LWS or not',
                        default=False,
                        type=bool)
    args = parser.parse_args()
    
    return args

def main():
    args = parse_args()
    
    logger, tb_logger, time_str = create_logger(
        args.out_dir, phase='test', create_tf_logs=True)
    
    logger.info(pprint.pformat(args))
    logger.info('Test ACE model')
    
    if args.model == 'Unet':
        model = ACE_Res50_UNet(num_classes=10, num_experts=args.experts, train_LWS = False, train_MLP=False, pretrained = True)
        if args.LWS:
            model = ACE_Res50_UNet(num_classes=10, train_LWS = True, num_experts=args.experts, pretrained = False)
            
    if args.model == 'Deeplabv3':
        model = deeplabv3P_resnet(num_classes=10, output_stride=8, pretrained_backbone=True, num_experts=args.experts)

    # Define loss function (criterion) and optimizer  
    device = torch.device("cuda")
    model = model.to(device)
    
    if args.experts==2:
        many_index = [0, 1, 5, 8, 9]
        few_index = [2, 3, 4, 6, 7]
        # many_index = [0, 1, 5, 7, 8, 9]
        # few_index = [2, 3, 4, 6]
        ls_index = [many_index, few_index]
        
    if args.experts==3:
        many_index = [0, 1, 5, 8, 9]
        medium_index = [6, 7]
        few_index = [2, 3, 4]
        ls_index = [many_index, medium_index, few_index]
        
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
    model.load_state_dict(bestmodel, strict=False)

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
    confusionMatrix = Oracale(test_loader, test_dataset, model, ls_index,
                         writer_dict, args)

    
    # np.save('/data/xiaolong/master_thesis/confusion_matrix/' + 'ACE_cm' + time_str, confusionMatrix)
    writer_dict['logger'].close()


def Oracale(test_loader, test_dataset, model, ls_index, writer_dict, args):
    logger = logging.getLogger(__name__)
    
    batch_time = AverageMeter()
    model.eval()
    
    metrics = MetricLogger(model.num_classes)
    device = torch.device("cuda")
    
    # generate a dict containing the name and parameters from 
    from collections import OrderedDict
    new_dict = OrderedDict()
    if args.model == 'Deeplabv3':
        for k, v in model.named_parameters():
            if k.startswith("classifier.SegHead"):
                new_dict[k] = v
    elif args.model =='Unet': 
        for k, v in model.named_parameters():
            if k.startswith("SegHead"):
                new_dict[k] = v
                
    if args.experts == 2:
        [many_index, few_index] = ls_index   
        if args.LWS:
            weight_many = new_dict['SegHead_many.conv2d.weight'].detach().cpu().numpy()
            weight_few = new_dict['SegHead_few.conv2d.weight'].detach().cpu().numpy()
        elif args.model=='Deeplabv3':
            weight_many = new_dict['classifier.SegHead_many.weight'].detach().cpu().numpy()
            weight_few = new_dict['classifier.SegHead_few.weight'].detach().cpu().numpy()
        elif args.model=='Unet':
            weight_many = new_dict['SegHead_many.weight'].detach().cpu().numpy()
            weight_few = new_dict['SegHead_few.weight'].detach().cpu().numpy()           

        weight_norm_many = LA.norm(weight_many, axis=1)
        weight_norm_few = LA.norm(weight_few, axis=1)
    
        f_scale = (np.mean(weight_norm_many) / np.mean(weight_norm_few[few_index,:]))
    
    with torch.no_grad():
        end = time.time()
        for i, (image, dem, mask) in enumerate(test_loader):
            
            # compute output
            image = image.to(device)
            dem = dem.to(device)
            input = torch.cat((image, dem), dim=1) #[B, 4, 200, 200]
            output = model(input)
            
            if args.experts == 2:
                [many_output, few_output], _ = output
                # new_few_output = few_output.clone()
                # new_few_output[:,many_index] = 0
                # # many_output[:, few_index] = 0
                # # few_output *= f_scale
                # final_output = many_output + new_few_output * f_scale
                # final_output[:,few_index] /= 2
                # final_output = torch.maximum(many_output, few_output)
            
            num_inputs = input.size(0) # B
            mask = mask.to(device) #[B, 1, 200, 200]
            few_mask = (mask >= 2) & (mask <= 4) | (mask == 6) | (mask == 7) #[B, 1, 200, 200]
            few_mask = few_mask.expand(few_output.size()) #[B, 10, 200, 200]
            many_mask = (few_mask == False)
            few_output = few_output * few_mask
            many_output = many_output * many_mask
            final_output = many_output + few_output

            # measure accuracy
            preds = get_final_preds(final_output.detach().cpu().numpy())
            # gt = torch.squeeze(mask).detach().cpu().numpy()
            gt = mask.squeeze(0).detach().cpu().numpy()
            metrics.update(gt, preds)
                
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % args.frequent == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                          i, len(test_loader), batch_time=batch_time)

                logger.info(msg)
                
            if writer_dict:
                writer = writer_dict['logger']
                global_steps = writer_dict['vis_global_steps']

                # Pick a random image in the batch to visualize
                idx = np.random.randint(0, num_inputs)

                # Unnormalize the image to [0, 255] to visualize
                input_image = image.detach().cpu().numpy()[idx]
                input_image = input_image * test_dataset.std.reshape(3,1,1) + test_dataset.mean.reshape(3,1,1)
                input_image[input_image > 1.0] = 1.0
                input_image[input_image < 0.0] = 0.0

                ## Turn the numerical labels into colorful map
                mask_image = mask.detach().cpu().numpy()[idx].astype(np.int64)
                mask_image = vis_seg_mask(mask_image.squeeze())

                final_output = torch.nn.functional.softmax(final_output, dim=1)
                output_mask = torch.argmax(final_output, dim=1, keepdim=False)

                output_mask = output_mask.detach().cpu().numpy()[idx]
                output_mask = vis_seg_mask(output_mask)

                writer.add_image('input_image', input_image, global_steps,
                    dataformats='CHW')
                writer.add_image('result_vis', output_mask, global_steps,
                    dataformats='HWC')
                writer.add_image('gt_mask', mask_image, global_steps,
                    dataformats='HWC')

                writer_dict['vis_global_steps'] = global_steps + 1         
            
        mean_cls, mean_iou, acc_cls, overall_acc = metrics.get_scores()
        confusionMatrix = metrics.get_confusion_matrix()
        acc_many, acc_medium, acc_few = metrics.get_acc_cat()
        
        logger.info('Mean IoU score: {:.3f}'.format(mean_iou))
        logger.info('Mean accuracy: {:.3f}'.format(mean_cls))
        logger.info('Overall accuracy: {:.3f}'.format(overall_acc))
        logger.info('Oracle accuracy: {Many:.3f}\t{Meidum:.3f}\t{Few:.3f}\t'.format(Many=acc_many,
                                                                               Meidum=acc_medium,
                                                                               Few=acc_few))
        classes = ["Background","Bedrock", "Bedrock with grass", "Large blocks", "Large blocks with grass", 
         "Scree", "Scree with grass", "Water area", "Forest", "Glacier"]
        for i in range(len(acc_cls)):
            logger.info(classes[i] + ' : {:.3f}'.format(acc_cls[i]))
            
    return confusionMatrix
            
if __name__ == '__main__':
    main()
