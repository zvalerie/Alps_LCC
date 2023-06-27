import os
import glob
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.segmentation.deeplabv3 import ASPP
from torchvision.models._utils import IntermediateLayerGetter   
from torch.linalg import vector_norm
from models_utils import _MultiExpertModel 
from ResNet import resnet50

class CNN_aggregator(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers = 256):
        super(CNN_aggregator, self).__init__()
        self.cnn =  nn.Sequential(
                            nn.Conv2d(input_dim, hidden_layers, 3, padding=1, bias=False),
                            nn.BatchNorm2d(hidden_layers),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(hidden_layers, output_dim, 1)
                        ) 
   
    def forward(self, x):
        return self.cnn(x) 

    
class DLV3P_w_BetterExperts(nn.Module):
    def __init__(self,  num_classes,  num_experts, use_lws=False, 
                 use_CNN_aggregator=False, use_MLP_aggregator  = False):
        
        super(DLV3P_w_BetterExperts, self).__init__()
       
        assert num_experts  in [2,3 ], 'Num_experts must be 2 or 3 ! '
        assert not ( use_MLP_aggregator and use_CNN_aggregator), 'only one of "use_MLP_aggregator" and "use_CNN_aggregator" can be True'
        self.num_experts = num_experts
        self.use_lws  = use_lws 
        
        in_channels = 2048
        low_level_channels = 256
        
        self.project = nn.Sequential( 
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        
        self.aspp = ASPP(in_channels, [12, 24, 36])

        if use_CNN_aggregator:
            self.classifier = CNN_aggregator(input_dim= num_experts * num_classes, output_dim= num_classes)    
            self.aggregation = 'cnn'
            
        elif use_MLP_aggregator :
            raise NotImplementedError
            self.aggregation = 'mlp'
        
        else:
            self.classifier = nn.Sequential( nn.ReLU(inplace=True))    # Useless layer too avoid empty and bug
            self.aggregation = None
        
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
               
       
        


    def forward(self, feature):
        low_level_feature = self.project( feature['low_level'] )
        output_feature = self.aspp(feature['out'])
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False)
        output_feature = torch.cat( [ low_level_feature, output_feature ], dim=1 )
        aggregator_output = None
            
        y_few = self.SegHead_few(output_feature)
        y_many = self.SegHead_many(output_feature)        
        if self.num_experts == 3:   
            y_medium = self.SegHead_medium(output_feature)
            
            
        if self.use_lws:                
            f_few = vector_norm( self.SegHead_few[3].weight.flatten()) / vector_norm(self.SegHead_many[3].weight.flatten())
            y_few = f_few * y_few
            if self.num_experts == 3: 
                f_medium = vector_norm( self.SegHead_medium[3] .weight.flatten()) / vector_norm(self.SegHead_many[3].weight.flatten())
                y_medium = f_medium * y_medium
            
        

        if self.num_experts == 3:       
            if self.aggregation is not None :
                
                expert_features = torch.cat([y_many, y_medium, y_few],dim=1) 
                aggregator_output = self.classifier(expert_features)
            return [y_many, y_medium, y_few], aggregator_output
        
        else :
            if self.aggregation is not None :
                
                expert_features = torch.cat([y_many,  y_few],dim=1) 
                aggregator_output = self.classifier(expert_features)
            
            return [y_many, y_few], aggregator_output

        
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



def ACE_deeplabv3P_w_Better_Experts(num_classes, num_experts,  pretrained_backbone=True,
                                    use_lws=False, use_CNN_aggregator=False, use_MLP_aggregator  = False):


    backbone = resnet50(
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=[False, True, True])
    
    inplanes = 2048
    low_level_planes = 256

    return_layers = {'layer4': 'out', 'layer1': 'low_level'}
    classifier = DLV3P_w_BetterExperts (
                                        num_classes, 
                                        num_experts, 
                                        use_lws=use_lws,
                                        use_CNN_aggregator = use_CNN_aggregator, 
                                        use_MLP_aggregator = use_MLP_aggregator,
                                        )
    
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    model = _MultiExpertModel(backbone, classifier, num_classes)
    # give a informative name :
    
    name = "ACE_deeplabv3P_w_Better_Experts"
    name += ' use_lws' if use_lws else ''
    name += ' use_CNN_aggregator' if use_CNN_aggregator else ''
    name += ' use_MLP_aggregator' if use_MLP_aggregator else ''
    model.__class__.__name__ = name
    return model


if __name__ == '__main__':
    
    x = torch.rand([64,4,20,20])
    model = ACE_deeplabv3P_w_Better_Experts( 
                                     num_classes=10, 
                                     pretrained_backbone=True, 
                                     num_experts=2,
                                     use_lws=True,
                                     use_MLP_aggregator=False,
                                     use_CNN_aggregator=True,
                                     )
    output =model(x) 
    for key in output.keys():
        print('output i=',key, output[key].shape)
        
