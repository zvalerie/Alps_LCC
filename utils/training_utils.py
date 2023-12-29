
import torch
import numpy as np
import random
import os 
import wandb 
from pprint import pprint
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset.SwissImageDataset import SwissImage
from dataset.flairDataset import FLAIRDataset
from utils.transforms import Compose, MyRandomRotation90, MyRandomHorizontalFlip, MyRandomVerticalFlip
from models import Res50_UNet, deeplabv3P_resnet, MCE_Unet
from models.models_utils import model_builder
  
from losses.ACE_losses import CELoss_2experts, CELoss_3experts, MyCrossEntropyLoss
from losses.ACE_losses import ClassBalancedLoss, WeightedCrossEntropyLoss
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
    if args.ds == 'TLM':
        alpha_2exp = 0.03
        alpha_3exp =(0.03,0.003)
    elif args.ds == 'FLAIR':
        alpha_2exp = 0.1781
        alpha_3exp =(0.1714,0.0067)
        
    if args.backbone == 'unet':
        if args.experts ==0  :
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay )
        elif args.experts ==2 :
            optimizer = optim.Adam([
                        {'params': model.backbone.parameters()},
                        {'params': model.expert_head.parameters()},
                        {'params': model.expert_tail.parameters(), 'lr' : args.lr *alpha_2exp}, 
                        ], lr= args.lr, weight_decay=args.weight_decay
                        )
        elif args.experts ==3 :
            optimizer = optim.Adam([
                {'params': model.backbone.parameters()},
                {'params': model.expert_head.parameters()},
                {'params': model.expert_body.parameters(), 'lr' : args.lr *alpha_3exp[0]}, 
                {'params': model.expert_tail.parameters(), 'lr' : args.lr *alpha_3exp[1]},  
                ], lr=args.lr,  weight_decay=args.weight_decay,
            )  
        
    elif args.backbone == 'deeplab':
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
                                    {'params': model.classifier.expert_tail.parameters(), 'lr' : args.lr *alpha_2exp}, 
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
                                    {'params': model.classifier.expert_body.parameters(), 'lr' : args.lr *alpha_3exp[0]}, 
                                    {'params': model.classifier.expert_tail.parameters(), 'lr' : args.lr *alpha_3exp[1]},  
                                    ], 
                                    lr=args.lr, 
                                    weight_decay=args.weight_decay,
                                )    
    
    if args.finetune_classifier_only  :
        optimizer = optim.SGD( model.classifier.classifier.parameters(),                                
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
        
    elif args.experts ==2 :    
        criterion = CELoss_2experts (args)
        
    elif args.experts == 3: 
        criterion = CELoss_3experts ( args)

    elif args.loss == 'seesaw' and args.experts == 0 :        
        criterion = SeesawLoss(num_classes= args.num_classes)
        
    elif args.loss == 'celoss' and args.experts == 0 :        
        criterion = MyCrossEntropyLoss(ignore_index=0)
    
    elif args.loss == 'cbloss' and args.experts == 0 :        
        criterion = ClassBalancedLoss(ignore_index=0, args=args)

    elif  args.finetune_classifier_only or 'merge' in args.aggregation:
        raise NotImplementedError
        assert args.aggregation != 'mean', 'No classifier to finetune in model ! '
        criterion = AggregatorLoss(args)
        
    elif 'moe' in args.aggregation :
        raise NotImplementedError
        criterion = AggregatorLoss(args)
        
    elif 'select' in args.aggregation :
        raise NotImplementedError
        criterion = selectExpertLoss(args)
        

    
    
    else:
        raise NotImplementedError

    
    return criterion


def get_model(args):
    """Get model based on arguments from config

    Args:
        args (dict) : args from config file
    """
   
    # Choose model : 
    if args.backbone =='deeplab':        
        if args.experts ==0 :
            model = deeplabv3P_resnet(num_classes=args.num_classes, output_stride=8, pretrained_backbone=True)
        
        elif args.experts == 2 or args.experts == 3 :
            model = model_builder (
                        num_classes = args.num_classes, 
                        num_experts = args.experts, 
                        use_lws = args.lws,
                        aggregation = args.aggregation,
                        )
    elif args.backbone =='unet':
        if args.experts ==0 :
            model = Res50_UNet(num_classes = args.num_classes,)
        elif args.experts == 2 or args.experts == 3 :
            model = MCE_Unet(num_classes = args.num_classes,
                             num_experts = args.experts,
                             use_lws = args.lws,
                             aggregation = args.aggregation,
                             )
        
    else :
        raise NotImplementedError 
    
   # Load weights from path:     
    if os.path.isfile(args.pretrained_weights): 
        checkpoint = torch.load(args.pretrained_weights)
        model.load_state_dict(checkpoint['state_dict'],strict=False)
        print('Model weights loaded from file:',args.pretrained_weights)
    else :
        print('Model trained from scratch')
        
    if args.finetune_classifier_only :      
        
        for name, param in model.named_parameters():
            if not ('classifier.classifier' in name ):
                param.requires_grad = False   
            else :  
                pass              
                #print('not frozen: ', name , param.requires_grad)      
        print('Model weights are frozen except for CNN or MLP layers')
        
        
        
    return model

def load_last_checkpoint (model,optimizer, args):
    
    last_model_path = os.path.join( args.out_dir, args.name,'last_model.pt')
        
    if not os.path.isfile(last_model_path):
        raise NameError ('Last model weights are not found in folder', args.out_dir , args.name )
    
    else :    
        checkpoint = torch.load(last_model_path)       
        best_weights = checkpoint['state_dict']
        model.load_state_dict (best_weights)
        
        state = checkpoint['optimizer']
        optimizer.load_state_dict(state)
        last_epoch = checkpoint['last_epoch']
        macc = checkpoint['perf']
        print('Catchup training. Using last model weights, last optimizer state, and start from epoch ', last_epoch )     
        
        return model, optimizer,last_epoch , macc
    
   
        
    
def get_FLAIR_dataloader(args=None, phase ='train'):
    
    data_dir = '/data/valerie/flair/'   # Modify the absolute path to the data here !  #TODO

    # Path to dataset splits : 
    test_csv = 'data/flair_split/tiny/test.csv'
    train_csv = 'data/flair_split/tiny/train.csv'
    val_csv =  'data/flair_split/tiny/val.csv'
    plot_csv = 'data/flair_split/tiny/plot.csv'
    patch_size = 512
    
    if  args.large_dataset:    
        train_csv = 'data/flair_split/base/train.csv'

    if args.debug :
        test_csv = 'data/flair_split/dev/test.csv'
        train_csv = 'data/flair_split/dev/train.csv'
        val_csv = 'data/flair_split/dev/val.csv'
        plot_csv = 'data/flair_split/dev/plot.csv'
        patch_size = 200
        
    
    if phase == 'train' :
        dataset_csv = train_csv
        patch_size = 400
    elif phase == 'val': 
        dataset_csv = val_csv
    elif phase == 'test': 
        dataset_csv = test_csv
    elif phase == 'plot':
        dataset_csv = plot_csv
    else :
        raise NotImplementedError
    
    dataset = FLAIRDataset(dataset_csv=dataset_csv,
                           data_dir=data_dir,
                           phase=phase,
                           patch_size=patch_size,
                           )
    
    loader = DataLoader(
        dataset= dataset,
        batch_size= args.bs ,
        shuffle= True if phase =='train' else False,
        num_workers= args.num_workers,
        pin_memory=True
        ) 
    return loader


def get_TLM_dataloader (args=None, phase ='train'):
    img_dir = '/data/valerie/rocky_tlm/rgb/' 
    dem_dir = '/data/valerie/rocky_tlm/dem/' 
    label_dir = '/data/valerie/master_Xiaolong/mask/'     

    # Path to dataset splits : 
    test_csv = 'data/split/test_dataset.csv'  # always the same test set    
    train_csv = 'data/split_subset/train_subset.csv'
    val_csv = 'data/split_subset/val_subset.csv'
    plot_csv = 'data/split/interesting_dataset_v2.csv'

    if  args.large_dataset:    
        train_csv = 'data/split/train_dataset.csv'
       # val_csv = 'data/split/val_dataset.csv'  

    common_transform = Compose([
        MyRandomHorizontalFlip(p=0.5),
        MyRandomVerticalFlip(p=0.5),
        MyRandomRotation90(p=0.5),
        ])

    img_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)
        ]) 

    if phase =='train' :
        dataset = SwissImage(
            train_csv, 
            img_dir, 
            dem_dir, 
            label_dir, 
            common_transform = common_transform, 
            img_transform = img_transform, 
            debug=args.debug)

    elif phase =='val':
        dataset = SwissImage(
            val_csv, 
            img_dir, 
            dem_dir, 
            label_dir, 
            common_transform=None,
            img_transform= None,
            debug=args.debug
            )
        
    elif phase == 'test':
        dataset = SwissImage(
            dataset_csv = test_csv,
            img_dir = img_dir,
            dem_dir = dem_dir,
            label_dir = label_dir,
            common_transform=None,
            img_transform= None,
            debug=args.debug,          
            )

    elif phase == 'plot':        
        dataset = SwissImage(
            dataset_csv = plot_csv,
            img_dir = img_dir,
            dem_dir = dem_dir,
            label_dir = label_dir,
            common_transform=None,
            img_transform= None,
            debug=args.debug,          
            )
    else :
        raise NotImplementedError
    
    loader = DataLoader(dataset= dataset,
        batch_size= args.bs if args is not None else 32,
        shuffle= True if phase =='train' else False,
        num_workers= args.num_workers if args is not None else 16,
        pin_memory=True
        )

    return loader

def get_dataloader(args=None, phase ='train'):
    """     
    Create training and validation datasets    based on arguments from config

    Args:
        args (dict) : args from config file
        phase (str) : indicates the phase in ['train','val','test]
    """    
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
    
    
    if args.ds == 'FLAIR':
        dl = get_FLAIR_dataloader(args=args,phase=phase)
        flair  =   {
        0:'Others',
        1: "building",      2: "pervious surface",      3: "impervious surface",    4: "bare soil",
        5: "water",         6: "coniferous",            7: "deciduous",             8: "brushwood",
        9: "vineyard",      10: "herbaceous vegetation",11: "agricultural land",    12: "plowed land",
        13: "swimming pool",        14: "snow",    15: "clear cut",    16: "mixed",
        17: "ligneous",    18: "greenhouse",
        }
        args.classes = {y:x for x,y in zip (flair.keys(),flair.values())}
   
    elif args.ds == 'TLM' :
        dl = get_TLM_dataloader(args=args, phase=phase)
        args.classes = {'Background':0, "Bedrock" : 1, "Bedrockwith grass" : 2,
                    "Large blocks" : 3, "Large blocks with grass" : 4, "Scree" : 5,
                    "Scree with grass" : 6,"Water" : 7,
                    "Forest" : 8, "Glacier" : 9, }
    
    else :
        raise NotImplementedError
    
    return dl


def setup_wandb_log(args):       
        
    # create new experiment in wandb
    if args.log_wandb :
        wandb.init(project = "ACE_revision", 
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
       

