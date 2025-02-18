import os
import glob
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.segmentation.deeplabv3 import ASPP
from torchvision.models._utils import IntermediateLayerGetter   
from torch.linalg import vector_norm

#from models.models_utils import _MultiExpertModel 
from models_utils import _MultiExpertModel
from ResNet import resnet50


    
class DeepLabHeadV3Plus_w_Experts(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, 
                 num_experts, aspp_dilate=[12, 24, 36], use_lws = False):
        super(DeepLabHeadV3Plus_w_Experts, self).__init__()
       
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
        
        self.num_experts = num_experts
        if num_experts == 2:
            self.SegHead_many = nn.Conv2d(256, num_classes, 1)
            self.SegHead_few = nn.Conv2d(256, num_classes, 1)
        elif num_experts == 3:
            self.SegHead_many = nn.Conv2d(256, num_classes, 1)
            self.SegHead_medium = nn.Conv2d(256, num_classes, 1)
            self.SegHead_few = nn.Conv2d(256, num_classes, 1)
        else:
            raise ValueError('num_experts must be 2 or 3')


        self._init_weight()
        
        self.use_lws  = use_lws
        

    def forward(self, feature):
        low_level_feature = self.project( feature['low_level'] )
        output_feature = self.aspp(feature['out'])
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False)
        final_feature = self.classifier( torch.cat( [ low_level_feature, output_feature ], dim=1 ) )
        
                
        if self.num_experts == 2:
            
            y_few = self.SegHead_few(final_feature)
            y_many = self.SegHead_many(final_feature)
            
            if self.use_lws:
                f_few = vector_norm(self.SegHead_many.weight.flatten()) / vector_norm( self.SegHead_few.weight.flatten())  
                y_few = f_few * y_few                



            return [y_many, y_few], None


        elif self.num_experts == 3:
            
            y_few = self.SegHead_few(final_feature)
            y_medium = self.SegHead_medium(final_feature)
            y_many = self.SegHead_many(final_feature)
            
            if self.use_lws:
                f_few = vector_norm(self.SegHead_many.weight.flatten())/ vector_norm( self.SegHead_few.weight.flatten()) 
                y_few = f_few * y_few
                
                f_medium = vector_norm(self.SegHead_many.weight.flatten()) /vector_norm( self.SegHead_medium .weight.flatten()) 
                y_medium = f_medium * y_medium
                
            
            return [y_many, y_medium, y_few], None

        
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



def ACE_deeplabv3P_w_Experts(num_classes, num_experts, output_stride=8, pretrained_backbone=True, use_lws=False):

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
    classifier =  DeepLabHeadV3Plus_w_Experts(inplanes, low_level_planes, num_classes, num_experts, aspp_dilate, use_lws=use_lws)
    
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    model = _MultiExpertModel(backbone, classifier, num_classes)
    model.__class__.__name__ = "ACE_deeplabv3P_w_Experts"
    return model


if __name__ == '__main__':
    
    x = torch.rand([64,4,20,20])
    model = ACE_deeplabv3P_w_Experts( 
                                     num_classes=10, 
                                     output_stride=8, 
                                     pretrained_backbone=True, 
                                     num_experts=3,
                                     use_lws =True,
                                     )
    output =model(x) 
    for key in output.keys():
        print('output i=',key, output[key].shape)
