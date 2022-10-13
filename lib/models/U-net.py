from pickle import TRUE
import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import resnet50

class DoubleConv (nn.Module):
    '''Conv2d + BN + ReLu'''
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.doubel_conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, 3),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.doubel_conv(x)
    
# class DownSampling(nn.Module):
#     '''Maxpool + DoubleConv. Please double the number of feature
#     channels at each downsampling step '''
#     def __init__(self, in_channel, out_channel):
#         super().__init__()
#         self.maxpool_D_conv = nn.Sequential(
#             nn.MaxPool2d(2),
#             DoubleConv(in_channel, out_channel)
#         )
        
#     def forward(self, x):
#         return self.maxpool_D_conv(x)
    
class Upsampling(nn.Module):
    '''ConvTransposed2d + Cropped feature map + DoubleConv'''
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channel, out_channel, 2, stride = 2)
        self.conv = DoubleConv(in_channel, out_channel)
        
    def forward(self, x1, x2):
        '''x2: cropped feature map'''
        x1 = self.up(x1)
        #  concatenation with cropped feature map
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class UNet(nn.Module):
    def __init__(self, num_classes, pretrained = TRUE, backbone = "resnet50"):
        super().__init__()
        out_channels = [64, 128, 256, 512]
        if backbone == "resnet50":
            self.resnet = resnet50(pretrained = pretrained)
            # skip_channels = [64, 256, 512, 1024, 2048]
            in_channels = [192, 512, 1024, 3072]
        
        self.up4 = Upsampling(in_channels[3], out_channels[3])
        self.up3 = Upsampling(in_channels[2], out_channels[2])
        self.up2 = Upsampling(in_channels[1], out_channels[1])
        self.up1 = Upsampling(in_channels[0], out_channels[0])
        self.segmentation_head = SegmentationHead(out_channels[0], num_classes)
        
        # self.outconv = nn.Conv2d(out_channels[0], num_classes, 1)
        
    def forward(self, x):
        if self.backbone == "resnet50" :
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(x)
        
        up4 = self.up4(feat4, feat5) 
        up3 = self.up3(feat3, up4) 
        up2 = self.up2(feat2, up3) 
        up1 = self.up1(feat1, up2) 
        y = self.segmentation_head(up1)
        return y

class SegmentationHead(nn.Sequential):
    '''set up a segmentation head'''
    def __init__(self, in_channels, out_channels, kernel_size = 3, upsampling = 1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = nn.Identity()
        super().__init__(conv2d, upsampling, activation)
        
        
        
        
        