import os
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from ResNet import resnet50

class DoubleConv (nn.Module):
    '''Conv2d + BN + ReLu'''
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.doubel_conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, 3,padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.doubel_conv(x)
    
    
class Upsampling(nn.Module):
    '''ConvTransposed2d + Cropped feature map + DoubleConv'''
    def __init__(self, in_channel, out_channel):
        super().__init__()
        # self.up = nn.ConvTranspose2d(in_channel, in_channel//2, 2, stride = 2)
        self.up = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.conv = DoubleConv(in_channel, out_channel)
        
    def forward(self, x1, x2):
        '''x2: cropped feature map'''
        #  concatenation with cropped feature map
        x2 = self.up(x2)
        if x2.size(3) != x1.size(3):
            x2 = x2[:,:,:x1.size(2), : x1.size(3)]
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x
    
class Res50_UNet(nn.Module):
    def __init__(self, num_classes, pretrained = True):
        super(Res50_UNet, self).__init__()
        out_channels = [64, 128, 256, 512]
        self.resnet = resnet50(pretrained = pretrained)
        in_channels = [192, 512, 1024, 3072]
        self.num_classes = num_classes
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
        self.outconv = nn.Conv2d(out_channels[0], self.num_classes, 1)
        
    def forward(self, x):
        [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(x)
        
        up4 = self.up4(feat4, feat5) 
        up3 = self.up3(feat3, up4) 
        up2 = self.up2(feat2, up3) 
        up1 = self.up1(feat1, up2) 
        y = self.up_conv(up1)
        y = self.outconv(y)
        return y

class SegmentationHead(nn.Sequential):
    '''set up a segmentation head'''
    def __init__(self, in_channels, out_channels, kernel_size = 3, upsampling = 1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = nn.Identity()
        super().__init__(conv2d, upsampling, activation)
        
 

if __name__ == '__main__':
    
    x = torch.rand([32,4,200,200])
    model = Res50_UNet (10,pretrained=True)
    output = model(x) 
    print('input shape',x.shape)
    print('output shape', len(output),output.shape)       
        
        