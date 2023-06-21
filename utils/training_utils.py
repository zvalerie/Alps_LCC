
import torch
import numpy as np
import random
import os 
import wandb 
from dataset.SwissImageDataset import SwissImage
from torch.utils.data import DataLoader
from utils.transforms import Compose, MyRandomRotation90, MyRandomHorizontalFlip, MyRandomVerticalFlip
from torchvision import transforms
from XL.lib.models.Unet import Res50_UNet
from XL.lib.models.DeepLabv3Plus import deeplabv3P_resnet
from XL.lib.core.loss import CrossEntropy2D, SeesawLoss, PixelPrototypeCELoss
from XL.lib.core.loss import ResCELoss, ResCELoss_3exp



def set_all_random_seeds(seed):
    """Set random seeds for reproductibility

    Args:
        seed (int): random number
    """
       
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True # True improve perf, False improve reproductibility
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed) 
    random.seed(seed) 
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    if  torch.initial_seed( ) != 2436  or  random.randint(0,10e6) != 1153963 :
        print( '!! Random seeds are NOT set correctly !! Please verify ')


def get_criterion (args):
    
    # Define loss function (criterion) and optimizer
    if args.loss == 'celoss':
        criterion = CrossEntropy2D(ignore_index=0)
    elif args.loss == 'seesawloss':
        criterion = SeesawLoss(ignore_index=0)
    elif args.model =='Deeplabv3_proto':
        criterion = PixelPrototypeCELoss()
        

    if args.experts == 3: 
        pass 
        many_index = [1, 5, 7, 8, 9]
        medium_index = [2, 6]
        few_index = [3, 4]
        criterion = ResCELoss_3exp(many_index, medium_index, few_index, args=args)
        lr_ratio = [0.03, 0.01] ## ratio of rare categories to frequent categories
    
    
    return criterion


def get_model(args):
   
    # Choose model : 
    if args.model == 'Unet':
        model = Res50_UNet(num_classes=10)
        
    elif args.model == 'Deeplabv3':
        model = deeplabv3P_resnet(num_classes=10, output_stride=8, pretrained_backbone=True)
        
    else:
       raise NotImplementedError
       
    
    return model


def get_dataloader(args=None, phase ='train'):
    """ 
    Create training and validation datasets    
    """    
   
    img_dir = '/home/valerie/data/rocky_tlm/rgb/' 
    dem_dir = '/home/valerie/data/rocky_tlm/dem/' 
    mask_dir = '/home/valerie/data/ace_alps/mask' 
    
    # Create output folder if needed :
    if args is not None and not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    if args is not None and not os.path.exists(os.path.join( args.out_dir, args.name)):
        os.mkdir(os.path.join( args.out_dir, args.name))
    
    # Path to dataset splits : 
    test_csv = 'data/split/test_dataset.csv'
    if args is not None and args.small_dataset:
        train_csv = 'data/split_subset/train_subset.csv'
        val_csv = 'data/split_subset/val_subset.csv'
    
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
    
    
    if phase =='train' :
        
        train_dataset = SwissImage(
            train_csv, 
            img_dir, 
            dem_dir, 
            mask_dir, 
            common_transform = common_transform, 
            img_transform = img_transform, 
            debug=args.debug)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size= args.bs if args is not None else 32,
            shuffle=True,
            num_workers= args.num_workers if args is not None else 16,
            pin_memory=True
        )
        
        return train_loader
    
    elif phase =='val':
        val_dataset = SwissImage(
            val_csv, 
            img_dir, 
            dem_dir, 
            mask_dir, 
            debug=args.debug)
        
        val_loader = DataLoader(
            val_dataset,
            batch_size= args.bs if args is not None else 32,
            shuffle=False,
            num_workers= args.num_workers if args is not None else 16,
            pin_memory=True
        )
    
        return val_loader
    
    elif phase == 'test':
        test_dataset = SwissImage(
            dataset_csv = test_csv,
            img_dir = img_dir,
            dem_dir = dem_dir,
            mask_dir = mask_dir,
            common_transform=None,
            img_transform= None,
            debug=args.debug,          
        )
        test_loader = DataLoader(test_dataset, 
            batch_size= args.bs if args is not None else 32,
            shuffle=False,
            num_workers= args.num_workers if args is not None else 16,
            pin_memory=True
        )
        return test_loader
    
    else :
        raise NotImplementedError
    

class AverageMeter(object):
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        

def setup_wandb_log(args):
    
    # create new experiment in wandb
    if args.log_wandb :
        wandb.init(project = "ACE_ALPS", 
                entity = "zvalerie",
                reinit = True,
                config = args,           
                )    
        
            
        wandb.run.name = args.name
        wandb.define_metric ('train_loss', summary = 'min'  )
        wandb.define_metric ('val_loss', summary = 'min' )
      #  wandb.define_metric ('train_accuracy' , summary = 'max')
        wandb.define_metric ('val_accuracy', summary = 'max')
        wandb.define_metric ('train_duration', summary = 'mean' )    
        wandb.define_metric ('val_duration', summary = 'mean' ) 
        wandb.define_metric ('val_mIoU', summary = 'max' )
     #   wandb.define_metric ('train_mIoU', summary = 'max' )
        wandb.define_metric ('lr', summary = 'last' )    

        
