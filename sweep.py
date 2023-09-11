
from utils.argparser import parse_args
import wandb

from train import main


def launch_sweep():
    wandb.init()
    args = parse_args()
    args.experts = 3
    args.debug = True
    args.epoch = 25 
    args.L2penalty = False
    args.aggregation ='MLP_merge'
    args.finetune_classifier_only =True
    args.log_wandb = True
    args.bs = 64
    args.pretrained_weights = '/home/valerie/project/Alps_LCC/out/MCE3_moptim/final.pt'

    
    print('wandb config', wandb.config)
    args.lr = wandb.config['lr']
    args.weight_decay = wandb.config['weight_decay']
    args.lws = wandb.config['lws']
    args.reweighted_aggregation = wandb.config['reweighted_aggregation']
    
    

    wandb.define_metric ('train_loss', summary = 'min'  )
    wandb.define_metric ('val_loss', summary = 'min' )
    wandb.define_metric ('val_accuracy', summary = 'max')
    wandb.define_metric ('train_duration', summary = 'mean' )    
    wandb.define_metric ('val_duration', summary = 'mean' ) 
    wandb.define_metric ('val_mIoU', summary = 'max' )
    wandb.define_metric ('lr', summary = 'last' )  
    main(args)
    
if __name__ == '__main__':
    

    wandb.login()
    
    sweep_configuration = {
        'method': 'bayes',
        'metric': 
        {
            'goal': 'maximize', 
            'name': 'test_macc',
            },
        'parameters': 
        {
            'lr': {'values': [ 0.0005, 0.00001  ,0.00005 , 0.00001 ,0.000005 ]}, 
            'weight_decay': {'values': [ 0.2, 0.1, 0.05, 0.01, 0.001 ]}, 
            'lws': {'values': [True, False]},   
            'reweighted_aggregation': {'values': ['CBL', 'None']},        
        }
    }
    
   
    # 3: Start the sweep
    sweep_id = wandb.sweep(
        sweep = sweep_configuration, 
        project='mlp_merge_sweep',
        entity = "zvalerie",
        )
    
    
    wandb.agent(sweep_id, function=launch_sweep, count=30)

