import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
class DownSampling(nn.Module):
    '''Maxpool + DoubleConv. Please double the number of feature
    channels at each downsampling step '''
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.maxpool_D_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channel, out_channel)
        )
        
    def forward(self, x):
        return self.maxpool_D_conv(x)
    
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
    def __init__(self, in_channels, n_classes):
        super().__init__()
        self.n_channels = in_channels
        self.inConv = DoubleConv(in_channels, 64) 
        self.down1 = DownSampling(64, 128)
        self.down2 = DownSampling(128, 256)
        self.down3 = DownSampling(256, 512)
        self.down4 = DownSampling(512, 1024)
        self.up1 = Upsampling(1024, 512)
        self.up2 = Upsampling(512, 256)
        self.up3 = Upsampling(256, 128)
        self.up4 = Upsampling(128, 64)
        self.outconv = nn.Conv2d(64, n_classes, 1)
        
    def forward(self, x):
        x1 = self.inConv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        y = self.up1(x5, x4)
        y = self.up2(y, x3)
        y = self.up3(y, x2)
        y = self.up4(y, x1)
        y = self.outconv(y)
        return y
        
        
        
        
        
        