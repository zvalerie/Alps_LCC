import os
import glob

import torch
from torch import nn
from torch.nn import functional as F
from XL.lib.models.ResNet import resnet50
from XL.lib.utils.utils import IntermediateLayerGetter
from XL.lib.models.DeepLabv3Plus import ASPP

class _SimpleSegmentationModel(nn.Module):
    def __init__(self, backbone, classifier, num_classes):
        super(_SimpleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.num_classes = num_classes
        
    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x,MLP_output = self.classifier(features)
        if isinstance(x, list):
            output = [F.interpolate(y, size=input_shape, mode='bilinear', align_corners=False) for y in x]
        else:
            output = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        
        if MLP_output != None:
            MLP_output = F.interpolate(MLP_output, size=input_shape, mode='bilinear', align_corners=False)
        return output, MLP_output
    
class DeepLabV3(_SimpleSegmentationModel):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.
    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    pass

class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, num_experts, aspp_dilate=[12, 24, 36], is_MLP=False):
        super(DeepLabHeadV3Plus, self).__init__()
        self.is_MLP = is_MLP
        self.num_experts = num_experts
        self.project = nn.Sequential( 
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        
        self.aspp = ASPP(in_channels, aspp_dilate)
        
        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        if num_experts == 2:
            self.SegHead_many = nn.Conv2d(256, num_classes, 1)
            self.SegHead_few = nn.Conv2d(256, num_classes, 1)
        elif num_experts == 3:
            self.SegHead_many = nn.Conv2d(256, num_classes, 1)
            self.SegHead_medium = nn.Conv2d(256, num_classes, 1)
            self.SegHead_few = nn.Conv2d(256, num_classes, 1)
        else:
            raise ValueError('num_experts must be 2 or 3')

        self.segHead = nn.Conv2d(256, num_classes, 1)
        
        if self.is_MLP:
            self.MLP = MLP(num_inputs=num_classes*num_experts, hidden_layers=128, num_outputs=num_experts)
        self._init_weight()

    def forward(self, feature):
        low_level_feature = self.project( feature['low_level'] )
        output_feature = self.aspp(feature['out'])
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False)
        final_feature = self.classifier( torch.cat( [ low_level_feature, output_feature ], dim=1 ) )
        # final_feature = torch.cat( [ low_level_feature, output_feature ], dim=1 )
        MLP_output = None
        if self.num_experts == 2:
            y_few = self.SegHead_few(final_feature)
            y_many = self.SegHead_many(final_feature)
            if self.is_MLP:
                y_stack = torch.cat((y_many, y_few), 1)
                y_stack = y_stack.permute(0, 2, 3, 1) #[B,H,W,20]
                exp_prob = self.MLP(y_stack) #[B,H,W,2]
                exp_prob = exp_prob.permute(0, 3, 1, 2)
                MLP_output = exp_prob
                # MLP_output = exp_prob[:,:1,:,:] * y_many + exp_prob[:,1:2,:,:] * y_few
            return [y_many, y_few], MLP_output

        elif self.num_experts == 3:
            y_few = self.SegHead_few(final_feature)
            y_medium = self.SegHead_medium(final_feature)
            y_many = self.SegHead_many(final_feature)
            if self.is_MLP:
                y_stack = torch.cat((y_many, y_medium, y_few), 1)
                y_stack = y_stack.permute(0, 2, 3, 1)
                exp_prob = self.MLP(y_stack)
                exp_prob = exp_prob.permute(0, 3, 1, 2)
                MLP_output = exp_prob
                # MLP_output = exp_prob[:,:1,:,:] * y_many + exp_prob[:,1:2,:,:] * y_medium + exp_prob[:,2:3,:,:] * y_few
            return [y_many, y_medium, y_few], MLP_output
        else:
            return self.segHead(final_feature)
        
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class MLP(nn.Module):
    def __init__(self, num_inputs, hidden_layers, num_outputs):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(num_inputs, hidden_layers),
                                 nn.ReLU(),
                                 nn.Linear(hidden_layers, num_outputs),
                                )
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        x = self.mlp(x)
        # x = self.softmax(x)
        return x

def ACE_deeplabv3P_resnet(num_classes, output_stride, pretrained_backbone, num_experts, is_MLP):

    if output_stride==8:
        replace_stride_with_dilation=[False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation=[False, False, True]
        aspp_dilate = [6, 12, 18]

    backbone = resnet50(
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation)
    
    inplanes = 2048
    low_level_planes = 256

    return_layers = {'layer4': 'out', 'layer1': 'low_level'}
    classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, num_experts, aspp_dilate, is_MLP)
    
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    model = DeepLabV3(backbone, classifier, num_classes)
    return model
