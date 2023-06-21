import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train image segmentation network')
    parser.add_argument('--lr',
                        help='learning rate',
                        default=1e-5,
                        type=float)
    parser.add_argument('--experts',
                        help='number of experts',
                        default=0,
                        type=int)
    parser.add_argument('--epoch',
                        help='training epoches',
                        default=50,
                        type=int)  
    parser.add_argument('--wd',
                        help='weight decay',
                        default=1e-2,
                        type=float)      
    parser.add_argument('--bs',
                        help='batch size',
                        default=32,
                        type=int)
    parser.add_argument('--lr_decay_rate',
                        help='scheduler_decay_rate',
                        default=0.1,
                        type=float)
    parser.add_argument('--loss',
                        help='which loss',
                        default='celoss',
                        type=str)
    # parser.add_argument('--patience',
    #                     help='scheduler_patience',
    #                     default=10,
    #                     type=int)
    parser.add_argument('--step_size',
                        help='step to decrease lr',
                        default = 10,
                        type=int)
    #parser.add_argument('--milestones',
    #                    help='step to decrease lr',
    #                    default = [10, 25],
    #                    type=list)
    parser.add_argument('--out_dir',
                        help='directory to save outputs',
                        default='out/train',
                        type=str)
    parser.add_argument('--model',
                        help='backbone of encoder',
                        default='Deeplabv3',
                        type=str)
    parser.add_argument('--frequent',
                        help='frequency of logging',
                        default=1,
                        type=int)
    # just an experience, the number of workers == cpu cores == 6 in this work station
    parser.add_argument('--num_workers',
                        help='num of dataloader workers',
                        default=6,
                        type=int)
    parser.add_argument('--continues',
                        help='continue training from checkpoint',
                        default=False,
                        type=bool)
    parser.add_argument('--debug',
                        help='is debuging?',
                        default=False,
                        type=bool)
    parser.add_argument('--small_dataset',
                        help='using small dataset for development',
                        default=True,
                        type=bool)
    parser.add_argument('--is_weighted_sampler',
                        help='is_weighted_sampler',
                        default=False,
                        type=bool)
    parser.add_argument('--MLP',
                        help='Train MLP layer to combine the results of experts',
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
    
    args = parser.parse_args()
    
    return args