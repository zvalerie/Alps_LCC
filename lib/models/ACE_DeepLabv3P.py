import os
import glob

import torch
from torch import nn
from torch.nn import functional as F
from lib.models.resnet import resnet50
from lib.utils.utils import IntermediateLayerGetter

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
        self.is_MLP = is_MLP
        # self.MLP = MLP(num_inputs=num_classes * num_experts, hidden_layers=128, output=3)
        self.MLP=MLP(20,128)
        self.num_experts = num_experts
        if num_experts == 2:
            self.SegHead_many = nn.Conv2d(256, num_classes, 1)
            self.SegHead_few = nn.Conv2d(256, num_classes, 1)
        elif num_experts == 3:
            self.SegHead_many = nn.Conv2d(256, num_classes, 1)
            self.SegHead_medium = nn.Conv2d(256, num_classes, 1)
            self.SegHead_few = nn.Conv2d(256, num_classes, 1)
        self.segHead = nn.Conv2d(256, num_classes, 1)
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
                MLP_output = exp_prob[:,:1,:,:] * y_many + exp_prob[:,1:2,:,:] * y_few
            return [y_many, y_few], MLP_output

        elif self.num_experts == 3:
            y_few = self.SegHead_few(final_feature)
            y_medium = self.SegHead_medium(final_feature)
            y_many = self.SegHead_many(final_feature)
            # if self.is_MLP:
            #     y_stack = torch.cat((y_many, y_medium, y_few), 1)
            #     y_stack = y_stack.permute(0, 2, 3, 1)
            #     exp_prob = self.MLP(y_stack)
            #     exp_prob = exp_prob.permute(0, 3, 1, 2)
            #     MLP_output = exp_prob
            #     # MLP_output = exp_prob[:,:1,:,:] * y_many + exp_prob[:,1:2,:,:] * y_medium + exp_prob[:,2:3,:,:] * y_few
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

class DeepLabHead(nn.Module):
    def __init__(self, in_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHead, self).__init__()

        self.classifier = nn.Sequential(
            ASPP(in_channels, aspp_dilate),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        return self.classifier( feature['out'] )

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class AtrousSeparableConvolution(nn.Module):
    """ 
    Atrous Separable Convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                            stride=1, padding=0, dilation=1, bias=True):
        super(AtrousSeparableConvolution, self).__init__()
        self.body = nn.Sequential(
            # Separable Conv
            nn.Conv2d( in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=in_channels ),
            # PointWise Conv
            nn.Conv2d( in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )
        
        self._init_weight()

    def forward(self, x):
        return self.body(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

class MLP(nn.Module):
    def __init__(self, num_inputs, hidden_layers, output):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(num_inputs, hidden_layers),
                                 nn.ReLU(),
                                 nn.Linear(hidden_layers, output),
                                )
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        x = self.mlp(x)
        x = self.softmax(x)
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

# if __name__ == '__main__':
#     model = ACE_deeplabv3P_resnet(10, 2, False, False)
#     for k, v in model.named_parameters():
#         # if k.startswith("SegHead"):
#             print(k)