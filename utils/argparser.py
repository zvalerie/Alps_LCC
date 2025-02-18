import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train image segmentation network')
    parser.add_argument('--lr',
                        help='learning rate',
                        default=1e-5,
                        type=float)
    parser.add_argument('--experts',
                        help='number of experts, between 2-3, 0 is baseline model',
                        default=0,
                        type=int)
    parser.add_argument('--epoch',
                        help='training epoches',
                        default=50,
                        type=int)  
    parser.add_argument('--bs',
                        help='batch size',
                        default=64,
                        type=int)
    parser.add_argument('--lr_decay_rate',
                        help='scheduler_decay_rate',
                        default=0.1,
                        type=float)
    parser.add_argument('--weight_decay',
                        help='weight decay',
                        default=1e-2,
                        type=float)
    parser.add_argument('--loss',
                        help='which loss for baseline model:celoss(default), seesaw, inverse_freq_weights',
                        default='celoss',
                        type=str)
    parser.add_argument('--step_size',
                        help='step to decrease lr',
                        default = 10,
                        type=int)
    parser.add_argument('--out_dir',
                        help='directory to save outputs',
                        default='out/august/',
                        type=str)
    parser.add_argument('--model',
                        help='model',
                        default='MCE',
                        type=str)
    parser.add_argument('--num_workers',
                        help='num of dataloader workers',
                        default=16,
                        type=int)
    parser.add_argument('--debug',
                        help='is debuging mode ?',
                        default=False,
                        type=bool)
    parser.add_argument('--large_dataset',
                        help='using large dataset for paper',
                        default=False,
                        type=bool)
    parser.add_argument('--force_cpu',
                        help='Device is set to cpu, no GPU usage.',
                        default=False,
                        type=bool)
    parser.add_argument('--seed',
                        help='random seed',
                        default=2436,
                        type=int)
    parser.add_argument('--name',
                        help='experiment name',
                        default='debug',
                        type=str)
    parser.add_argument('--log_wandb',
                        help='log experiment to wandb',
                        default=False,
                        type=bool)  
    parser.add_argument('--test_only',
                        help='only run testing',
                        default=False,
                        type=bool) 
    parser.add_argument('--L2penalty',
                        help='use a complementary loss term as penalty for predicting non target classes',
                        default=False,
                        type=bool) 
    parser.add_argument('--lws',
                        help='use a learnable weights scaling',
                        default=False,
                        type=bool) 
    parser.add_argument('--separate_backprop',
                        help='use a separate backprop for each expert, E1 update the backbone',
                        default=False,
                        type=bool)
    parser.add_argument('--aggregation',
                        help='method used for aggregation choose from :mean,CNN_merge,MLP_merge,CNN_select,MLP_select,max_pool,MLP_moe',
                        default='mean',
                        type=str)
    parser.add_argument('--finetune_classifier_only',
                        help='Train MLP/CNN only, freeze Deeplabv3 backbone',
                        default=False,
                        type=bool)
    parser.add_argument('--reweighted_aggregation',
                        help='give weights to classes during the training of aggregation network: None, inverse_frequency, CBL',
                        default='None',
                        type=str)
    parser.add_argument('--zero_nontarget_expert',
                        help='Zero the prediction of expert on non target classes',
                        default=False,
                        type=bool)
    parser.add_argument('--not_adaptive_lr',
                        help='use the same lr for all experts, default : False (use adaptative lr)',
                        default=False,
                        type=bool)
    parser.add_argument('--catchup_training',
                        help='catch up the training if it was interrupted',
                        default=False,
                        type=bool)
    parser.add_argument('--pretrained_weights',
                        help='pretrained weights',
                        default='Nope',
                        type=str)
    args = parser.parse_args()
    
    return args