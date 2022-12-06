import os
import logging
import time

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth.tar'):
    """Save model checkpoint

    Args:
        states: model states.
        is_best (bool): whether to save this model as best model so far.
        output_dir (str): output directory to save the checkpoint
        filename (str): checkpoint name
    """
    torch.save(states, os.path.join(output_dir, filename))
    if is_best and 'state_dict' in states:
        torch.save(states['state_dict'],
                   os.path.join(output_dir, 'model_best.pth.tar'))

def create_logger(out_dir, phase='train', create_tf_logs=True):
    """Create text logger and TensorBoard writer objects

    Args:
        out_dir (str): output directory for saving logs.
        phase (str): short description for log, will be appended to log filename.
        create_tf_logs (bool): whether to create TensorBoard writer or not
    Returns:
        logger: text logger
        writer: TensorBoard writer
    """
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}.log'.format(time_str, phase)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    final_log_file = os.path.join(out_dir, log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    if create_tf_logs:
        try:
            from tensorboardX import SummaryWriter
            writer = SummaryWriter(os.path.join(out_dir, time_str))
        except:
            writer = None
    else:
        writer = None

    return logger, writer, time_str

def get_optimizer(model, type, args, lr_ratio):
    base_lr = args.lr
    few_params = list(map(id, model.SegHead_few.parameters())) 
    many_paramas = filter(lambda p : id(p) not in few_params, model.parameters())
    params = [
            {"params": many_paramas, "lr": args.lr},
            {"params": model.SegHead_few.parameters(), "lr": args.lr * lr_ratio},
    ]
    if type == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=base_lr,
            momentum=0.9,
            weight_decay=1e-4,
            nesterov=True,
        )
    elif type == "ADAM":
        optimizer = torch.optim.Adam(
            params,
            lr=base_lr,
            betas=(0.9, 0.999),
            weight_decay=args.wd,
        )
    else:
        raise NotImplementedError
    return optimizer

def get_scheduler(optimizer, type, args):
    if type == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            args.step_size,
            gamma=args.lr_decay_rate,
        )
    # elif type == "warmup":
    #     scheduler = WarmupMultiStepLR(
    #         optimizer,
    #         args.step_size,
    #         gamma=args.lr_decay_rate,
    #         warmup_epochs=args.warm_epoch,
    #     )
    else:
        raise NotImplementedError("Unsupported LR Scheduler: {}".format(type))

    return scheduler

def get_category_list(label_path, csv_path):
    df = pd.read_csv(csv_path)
    filename = df['mask']
    class_dict = dict()
    
    for i in tqdm(range(filename.shape[0])):
        try:
            img = Image.open(label_path + '/' + filename[i])
        except:
            pass
            continue
        img = np.array(img)
        classes, counts = np.unique(img, return_counts=True)
        for i in range(len(classes)):
            class_dict[classes[i]]=class_dict.get(classes[i],0)+counts[i]
    
    print(class_dict)
    f = open('number per class.txt', 'w')
    f.write(str(class_dict))
    f.close()
    
   
if __name__ == '__main__':
    label_path = '/data/xiaolong/mask'
    label_csv_path = '/data/xiaolong/master_thesis/data_preprocessing/label_selection_0.1_rgb.csv'
    get_category_list(label_path, label_csv_path)
