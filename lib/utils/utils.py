import os
import logging
import time

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
from collections import OrderedDict

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

def get_optimizer(model, type, num_experts, base_lr, lr_ratio):
    if num_experts == 2:
        few_params = list(map(id, model.classifier.SegHead_few.parameters())) 
        many_paramas = filter(lambda p : id(p) not in few_params, model.parameters())
        params = [
                {"params": many_paramas, "lr": base_lr},
                {"params": model.classifier.SegHead_few.parameters(), "lr": base_lr * lr_ratio[0]},
        ]
    
    if num_experts == 3:
        few_params = list(map(id, model.SegHead_few.parameters())) 
        medium_params = list(map(id, model.SegHead_medium.parameters())) 
        many_paramas = filter(lambda p : id(p) not in few_params + medium_params, model.parameters())
        params = [
                {"params": many_paramas, "lr": base_lr},
                {"params": model.SegHead_medium.parameters(), "lr": base_lr * lr_ratio[0]},
                {"params": model.SegHead_few.parameters(), "lr": base_lr * lr_ratio[1]},
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
            weight_decay=1e-2,
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

def get_lr_ratio(label_path, csv_path, expert_idx):
    '''
    expert_idx: list, [[many],[medium],[few]]
    '''
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

    # if expert_idx != None:
    #     num_many = 0 
    #     num_medium =0
    #     num_few =0
    #     for k, v in class_dict.items():
    #         if k == 0:
    #             continue
    #         num_many += v
    #         if k not in expert_idx[0]:
    #             num_medium += v
    #         # if k in expert_idx[2]:
    #         #     num_few += v
    #     medium_ratio = num_medium/num_many
    #     few_ratio = num_few/num_many
    #     print(medium_ratio, few_ratio)
    # return medium_ratio, few_ratio
    
    # f = open('number per class.txt', 'w')
    # f.write(str(class_dict))
    # f.close()
    print(class_dict)

class _SimpleSegmentationModel(nn.Module):
    def __init__(self, backbone, classifier, num_classes):
        super(_SimpleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.num_classes = num_classes
        
    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x = self.classifier(features)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x

class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model
    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.
    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.
    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    Examples::
        >>> m = torchvision.models.resnet18(pretrained=True)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """
    def __init__(self, model, return_layers, hrnet_flag=False):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        self.hrnet_flag = hrnet_flag

        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():
            if self.hrnet_flag and name.startswith('transition'): # if using hrnet, you need to take care of transition
                if name == 'transition1': # in transition1, you need to split the module to two streams first
                    x = [trans(x) for trans in module]
                else: # all other transition is just an extra one stream split
                    x.append(module(x[-1]))
            else: # other models (ex:resnet,mobilenet) are convolutions in series.
                x = module(x)

            if name in self.return_layers:
                out_name = self.return_layers[name]
                if name == 'stage4' and self.hrnet_flag: # In HRNetV2, we upsample and concat all outputs streams together
                    output_h, output_w = x[0].size(2), x[0].size(3)  # Upsample to size of highest resolution stream
                    x1 = F.interpolate(x[1], size=(output_h, output_w), mode='bilinear', align_corners=False)
                    x2 = F.interpolate(x[2], size=(output_h, output_w), mode='bilinear', align_corners=False)
                    x3 = F.interpolate(x[3], size=(output_h, output_w), mode='bilinear', align_corners=False)
                    x = torch.cat([x[0], x1, x2, x3], dim=1)
                    out[out_name] = x
                else:
                    out[out_name] = x
        return out

# if __name__ == '__main__':
#     label_path = '/data/xiaolong/mask'
#     label_csv_path = '/data/xiaolong/master_thesis/data_preprocessing/subset/val_subset.csv'
#     expert_idx=[[1, 5, 8, 9], [2, 3, 4, 6, 7]]
#     get_lr_ratio(label_path, label_csv_path, expert_idx)
