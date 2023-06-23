import os
import glob
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.segmentation.deeplabv3 import ASPP
from torchvision.models._utils import IntermediateLayerGetter   

from models_utils import _MultiExpertModel 
from ResNet import resnet50


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
    
    

    
class DLV3P_w_BetterExperts(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, 
                 num_experts, aspp_dilate=[12, 24, 36], is_MLP=False):
        super(DLV3P_w_BetterExperts, self).__init__()
       
        self.project = nn.Sequential( 
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        
        self.aspp = ASPP(in_channels, aspp_dilate)

        self.classifier = nn.Sequential( nn.ReLU(inplace=True))    # Useless layer too avoid empty and bug
        
        self.num_experts = num_experts
        if num_experts ==2 or num_experts == 3:
            
            self.SegHead_many = nn.Sequential(
                            nn.Conv2d(304, 256, 3, padding=1, bias=False),
                            nn.BatchNorm2d(256),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(256, num_classes, 1)
                        ) 
            self.SegHead_few = nn.Sequential(
                                nn.Conv2d(304, 256, 3, padding=1, bias=False),
                                nn.BatchNorm2d(256),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(256, num_classes, 1)
                            )           
                     
            if num_experts == 3:
                self.SegHead_medium = nn.Sequential(
                                nn.Conv2d(304, 256, 3, padding=1, bias=False),
                                nn.BatchNorm2d(256),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(256, num_classes, 1)
                            )
            
        else:
            raise ValueError('num_experts must be 2 or 3')

        self.is_MLP = is_MLP
        if self.is_MLP:
            self.MLP = MLP(num_inputs=num_classes*num_experts, hidden_layers=128, num_outputs=num_experts)
        self._init_weight()

    def forward(self, feature):
        low_level_feature = self.project( feature['low_level'] )
        output_feature = self.aspp(feature['out'])
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False)
        output_feature = torch.cat( [ low_level_feature, output_feature ], dim=1 )
        
        MLP_output = None
        if self.num_experts == 2:
            y_few = self.SegHead_few(output_feature)
            y_many = self.SegHead_many(output_feature)
            if self.is_MLP:
                raise NotImplementedError
                y_stack = torch.cat((y_many, y_few), 1)
                y_stack = y_stack.permute(0, 2, 3, 1) #[B,H,W,20]
                exp_prob = self.MLP(y_stack) #[B,H,W,2]
                exp_prob = exp_prob.permute(0, 3, 1, 2)
                MLP_output = exp_prob
                # MLP_output = exp_prob[:,:1,:,:] * y_many + exp_prob[:,1:2,:,:] * y_few
            return [y_many, y_few], MLP_output

        elif self.num_experts == 3:
            y_few = self.SegHead_few(output_feature)
            y_medium = self.SegHead_medium(output_feature)
            y_many = self.SegHead_many(output_feature)
            if self.is_MLP:
                raise NotImplementedError
                y_stack = torch.cat((y_many, y_medium, y_few), 1)
                y_stack = y_stack.permute(0, 2, 3, 1)
                exp_prob = self.MLP(y_stack)
                exp_prob = exp_prob.permute(0, 3, 1, 2)
                MLP_output = exp_prob
                # MLP_output = exp_prob[:,:1,:,:] * y_many + exp_prob[:,1:2,:,:] * y_medium + exp_prob[:,2:3,:,:] * y_few
            return [y_many, y_medium, y_few], MLP_output

        
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



def ACE_deeplabv3P_w_Better_Experts(num_classes, num_experts, is_MLP ,output_stride=8, pretrained_backbone=True, ):

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
    classifier = DLV3P_w_BetterExperts (inplanes, low_level_planes, num_classes, num_experts, aspp_dilate, is_MLP)
    
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    model = _MultiExpertModel(backbone, classifier, num_classes)
    model.__class__.__name__ = "ACE_deeplabv3P_w_Better_Experts"
    return model


if __name__ == '__main__':
    
    x = torch.rand([64,4,200,200])
    model = ACE_deeplabv3P_w_Better_Experts( 
                                     num_classes=10, 
                                     output_stride=8, 
                                     pretrained_backbone=True, 
                                     num_experts=2,
                                     is_MLP=False)
    output =model(x) 
    for key in output.keys():
        print('output i=',key, output[key].shape)
        
