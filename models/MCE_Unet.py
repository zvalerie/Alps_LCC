import os
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from Unet import Res50_UNet
from MCE_model import Expert
from torch.linalg import vector_norm


class MCE_Unet(nn.Module):
    def __init__(self,  num_classes,  num_experts, use_lws=False, 
                 aggregation='mean'):
        
        super(MCE_Unet, self).__init__()
        
        assert num_experts  in [2,3 ], 'Num_experts must be 2 or 3 ! '
        assert aggregation  in [ 'mean', 'max_pool'] , 'other aggregation not implemented'
        self.num_experts = num_experts
        self.use_lws  = use_lws 
        self.num_classes=num_classes
        self.aggregation = aggregation
        self.backbone = Res50_UNet(num_classes=num_classes,pretrained=True)
        self.backbone.outconv = nn.Identity() # Replace the Unet classifier by identity (i.e. do nothing)
        
        

        self.expert_head = Expert(input_dim=64, output_dim=num_classes) 
        self.expert_tail = Expert(input_dim=64, output_dim=num_classes)          
        if num_experts == 3:
            self.expert_body = Expert(input_dim=64, output_dim=num_classes)       
        
        # give an informative name :
        name = "MCE_UNET"
        name += ' use_lws' if use_lws else ''
        self.__name__ = name


    def forward(self, input):
        # DeepLabV3 backbone : 
       
        output_feature =   self.backbone(input)    
        
        # Experts :
        y_tail = self.expert_tail(output_feature)
        y_head = self.expert_head(output_feature)        
        if self.num_experts == 3:   
            y_body = self.expert_body(output_feature)
        
        if self.use_lws :       
            f_tail= vector_norm(self.expert_head.cnn2.weight.flatten()) / vector_norm( self.expert_tail.cnn2.weight.flatten()) 
            y_tail= f_tail* y_tail
            if self.num_experts == 3: 
                f_body = vector_norm(self.expert_head.cnn2.weight.flatten()) / vector_norm( self.expert_body.cnn2.weight.flatten()) 
                y_body = f_body * y_body
                
        if self.num_experts == 2 :
            output = {'exp_0':y_head, 'exp_1':y_tail} 
        else:   
            output = {'exp_0':y_head, 'exp_1':y_body,'exp_2':y_tail}
        
        return output





if __name__ == '__main__':
    
    x = torch.rand([8,4,200,200])
    model = MCE_Unet(num_classes=19,
                     num_experts=3,
                     use_lws=True)
                
    output = model(x) 
    for key in output.keys():
        print('output i=',key, output[key].shape)
        
