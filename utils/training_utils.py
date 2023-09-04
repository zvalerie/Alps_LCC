
import torch
import numpy as np
import random
import os 
import wandb 
from pprint import pprint
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset.SwissImageDataset import SwissImage
from utils.transforms import Compose, MyRandomRotation90, MyRandomHorizontalFlip, MyRandomVerticalFlip
from models import Res50_UNet, deeplabv3P_resnet
from models.models_utils import model_builder
  
from losses.ACE_losses import CELoss_2experts, CELoss_3experts, MyCrossEntropyLoss, WeightedCrossEntropyLoss
from losses.aggregator_losses import AggregatorLoss
from losses.selectExpertLoss import selectExpertLoss
from losses.SeesawLoss import SeesawLoss
from torch import optim


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

def get_optimizer (model,args):
    """Set optimizerbased on arguments from config

    Args:
        model (nn.Module): model used in the main loop
        args (dict) : args from config file
    """
    
    
    if args.experts ==0  :
        optimizer = optim.Adam(model.parameters(), 
                               lr=args.lr, 
                               weight_decay=args.weight_decay
                               )
        
    elif args.experts ==2 :
       
       
        optimizer = optim.Adam([
                                {'params': model.backbone.parameters()},
                                {'params': model.classifier.project.parameters()},
                                {'params': model.classifier.aspp.parameters()},
                                {'params': model.classifier.classifier.parameters()},
                                {'params': model.classifier.expert_head.parameters()},
                                {'params': model.classifier.expert_tail.parameters(), 'lr' : args.lr *0.03}, 
                                ], 
                                lr= args.lr, 
                                weight_decay=args.weight_decay
                                )
        
    elif args.experts ==3 :

        optimizer = optim.Adam(
                                [
                                {'params': model.backbone.parameters()},
                                {'params': model.classifier.project.parameters()},
                                {'params': model.classifier.aspp.parameters()},
                                {'params': model.classifier.classifier.parameters()},
                                {'params': model.classifier.expert_head.parameters()},
                                {'params': model.classifier.expert_body.parameters(), 'lr' : args.lr *0.03}, 
                                {'params': model.classifier.expert_tail.parameters(), 'lr' : args.lr *0.003},  
                                ], 
                                lr=args.lr, 
                                weight_decay=args.weight_decay,
                               )          
    
    if args.not_adaptive_lr :
        optimizer = optim.Adam(model.parameters(), 
                               lr=args.lr, 
                               weight_decay=args.weight_decay
                               )
        print('--- use uniform lr for all layers (not adaptive lr)---')
        if False :
            for param_group in optimizer.param_groups:
                print(f"Learning rate for param group ", param_group['lr'])
                            
    
    return optimizer


def get_criterion (args):
    """Set criterion based on arguments from config

    Args:
        args (dict) : args from config file
    """
    
    # Define loss function (criterion) and optimizer
    if args.loss == 'inverse_freq_weights':        
        criterion = WeightedCrossEntropyLoss(ignore_index=0,args=args)

    elif args.loss == 'seesaw' and args.experts == 0 :        
        criterion = SeesawLoss(num_classes= 10)
        
    elif args.loss == 'celoss' and args.experts == 0 :        
        criterion = MyCrossEntropyLoss(ignore_index=0)

    elif  args.finetune_classifier_only or 'merge' in args.aggregation:
        assert args.aggregation != 'mean', 'No classifier to finetune in model ! '
        criterion = AggregatorLoss(args)
        
    elif 'select' in args.aggregation :
        criterion = selectExpertLoss(args)
        
    elif args.experts ==2 :    
        criterion = CELoss_2experts (args)
        
    elif args.experts == 3: 
        criterion = CELoss_3experts ( args)
    
    
    else:
        raise NotImplementedError

    
    return criterion


def get_model(args):
    """Get model based on arguments from config

    Args:
        args (dict) : args from config file
    """
   
    # Choose model : 

    if args.experts ==0 :
        model = deeplabv3P_resnet(num_classes=10, output_stride=8, pretrained_backbone=True)
    
    elif args.experts == 2 or args.experts == 3 :
        model = model_builder (
                    num_classes = 10, 
                    num_experts = args.experts, 
                    use_lws = args.lws,
                    aggregation = args.aggregation,
                    )
    else :
        raise NotImplementedError
   
    if os.path.isfile(args.pretrained_weights): 
        checkpoint = torch.load(args.pretrained_weights)
        model.load_state_dict(checkpoint['state_dict'],strict=False)
        print('Model weights loaded from file:',args.pretrained_weights)
        
    if args.finetune_classifier_only :      
        
        for name, param in model.named_parameters():
            if not ('classifier.classifier' in name ):
                param.requires_grad = False   
            else : 
                pass
                #print('not frozen ', name , param.requires_grad)      
        print('Model weights are frozen except for CNN or MLP layers')
    
       
    
    return model



def get_dataloader(args=None, phase ='train'):
    """     Create training and validation datasets    based on arguments from config

    Args:
        args (dict) : args from config file
        phase (str) : indicates the phase in ['train','val','test]
    """    
   
    img_dir = '/home/valerie/data/rocky_tlm/rgb/' 
    dem_dir = '/home/valerie/data/rocky_tlm/dem/' 
    label_dir = '/home/valerie/data/ace_alps/mask' 
    
    # Create output folder if needed :
    if args is not None and not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    if args is not None and not os.path.exists(os.path.join( args.out_dir, args.name)):
        os.mkdir(os.path.join( args.out_dir, args.name))
        
    # Save config in file
    import json
    fn = os.path.join( args.out_dir, args.name,'config.json')
    with open(fn, 'w') as file:
        json.dump(vars(args), file, indent=4)

    
    # Path to dataset splits : 
    test_csv = 'data/split/test_dataset.csv'  # always the same test set    
    train_csv = 'data/split_subset/train_subset.csv'
    val_csv = 'data/split_subset/val_subset.csv'
    
    if  args.large_dataset:    
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
            label_dir, 
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
            label_dir, 
            common_transform=None,
            img_transform= None,
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
            label_dir = label_dir,
            common_transform=None,
            img_transform= None,
            debug=args.debug,          
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size= args.bs if args is not None else 32,
            shuffle=False,
            num_workers= args.num_workers if args is not None else 16,
            pin_memory=True
        )
        return test_loader
    
    elif phase == 'plot':
        plot_csv = '/home/valerie/Projects/Alps_LCC/data/split/interesting_dataset.csv'
        test_dataset = SwissImage(
            dataset_csv = plot_csv,
            img_dir = img_dir,
            dem_dir = dem_dir,
            label_dir = label_dir,
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
    



def setup_wandb_log(args):
        
    

        
        
    # create new experiment in wandb
    if args.log_wandb :
        wandb.init(project = "ACE_ALPS_Au18", 
                entity = "zvalerie",
                reinit = True,
                config = args,           
                )    
        
            
        wandb.run.name = args.name
        wandb.define_metric ('train_loss', summary = 'min'  )
        wandb.define_metric ('val_loss', summary = 'min' )
        wandb.define_metric ('val_accuracy', summary = 'max')
        wandb.define_metric ('train_duration', summary = 'mean' )    
        wandb.define_metric ('val_duration', summary = 'mean' ) 
        wandb.define_metric ('val_mIoU', summary = 'max' )
        wandb.define_metric ('lr', summary = 'last' )    
       

