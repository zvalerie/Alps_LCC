import os
import glob

import math

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from typing import Type, List, Optional
from XL.lib.models.Unet import Upsampling
from XL.lib.models.ResNet import resnet50
model_urls = {'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth'}

class ACE_Res50_UNet(nn.Module):
    def __init__(self, num_classes, num_experts, is_MLP, pretrained = True):
        super(ACE_Res50_UNet, self).__init__()
        out_channels = [64, 128, 256, 512]
        self.resnet = resnet50(pretrained = pretrained)
        in_channels = [192, 512, 1024, 3072]
        self.num_classes = num_classes
        self.num_experts = num_experts
        self.is_MLP = is_MLP
        self.up4 = Upsampling(in_channels[3], out_channels[3])
        self.up3 = Upsampling(in_channels[2], out_channels[2])
        self.up2 = Upsampling(in_channels[1], out_channels[1])
        self.up1 = Upsampling(in_channels[0], out_channels[0])
        self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor = 2), 
                nn.Conv2d(out_channels[0], out_channels[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
                nn.Conv2d(out_channels[0], out_channels[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
            )
        if num_experts == 2:      
            self.SegHead_many = nn.Conv2d(out_channels[0], self.num_classes, 1)
            self.SegHead_few = nn.Conv2d(out_channels[0], self.num_classes, 1)
        elif num_experts == 3:
            self.SegHead_many = nn.Conv2d(out_channels[0], self.num_classes, 1)
            self.SegHead_medium = nn.Conv2d(out_channels[0], self.num_classes, 1)
            self.SegHead_few = nn.Conv2d(out_channels[0], self.num_classes, 1)
        else:
            raise NotImplementedError
        if self.is_MLP:
            self.MLP = MLP(num_inputs=num_classes*num_experts, hidden_layers=128, num_outputs=num_experts)
        
    def forward(self, x):
        [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(x)
        
        up4 = self.up4(feat4, feat5) 
        up3 = self.up3(feat3, up4) 
        up2 = self.up2(feat2, up3) 
        up1 = self.up1(feat1, up2) 
        y = self.up_conv(up1)
        if self.num_experts == 2:
            y_many = self.SegHead_many(y)
            y_few = self.SegHead_few(y) 
            MLP_output = None
            if self.is_MLP==True:
                y_stack = torch.cat((y_many, y_few), 1)
                y_stack = y_stack.permute(0, 2, 3, 1) #[B,H,W,20]
                exp_prob = self.MLP(y_stack) #[B,H,W,2]
                exp_prob = exp_prob.permute(0, 3, 1, 2) #[B,2,H,W]
                # MLP_output = exp_prob
                MLP_output = exp_prob[:,:1,:,:] * y_many + exp_prob[:,1:2,:,:] * y_few
            return [y_many, y_few], MLP_output
        
        elif self.num_experts == 3:
            y_few = self.SegHead_few(y)
            y_medium = self.SegHead_medium(y)
            y_many = self.SegHead_many(y)
            return [y_many, y_medium, y_few], None
        
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
        x = self.softmax(x)
        return x
    