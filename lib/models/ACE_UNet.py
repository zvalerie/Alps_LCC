import os
import glob

import math

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from typing import Type, List, Optional

model_urls = {'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth'}

def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    ''' 3 x 3 convolution '''
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    ''' 1 x 1 convolution'''
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# Bottleneck for resnet50
class Bottleneck(nn.Module) :
    expansion = 4
    
    def __init__(self, inplanes, outplanes, stride=1, downsample=None,
                 dilation=1, norm_layer=nn.BatchNorm2d):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, outplanes)
        self.bn1 = norm_layer(outplanes)
        self.conv2 = conv3x3(outplanes, outplanes, stride, dilation)
        self.bn2 = norm_layer(outplanes)
        self.conv3 = conv1x1(outplanes, outplanes * Bottleneck.expansion)
        self.bn3 = norm_layer(outplanes * Bottleneck.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
class ResNet50(nn.Module):
    def __init__(self, 
                 block = Bottleneck,
                 layers = [3, 4, 6, 3],
                 in_channels = 4,
                 out_channels = [64, 128, 256, 512],
                 norm_layer = nn.BatchNorm2d
                ):
        super(ResNet50, self).__init__()
        
        self.in_channels = in_channels 
        self._out_channels = out_channels
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        
        # input layer
        # 200, 200, 4 -> 100, 100, 64
        self.conv1 = nn.Conv2d(4, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # 100, 100, 64 -> 50, 50, 64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        #conv layers
        # 50, 50, 64 -> 50, 50, 256
        self.layer1 = self._make_layer(block, out_channels[0], layers[0])
        # 50, 50, 256 -> 25, 25, 512
        self.layer2 = self._make_layer(block, out_channels[1], layers[1], stride=2)
        #25, 25, 512 -> 13, 13, 1024
        self.layer3 = self._make_layer(block, out_channels[2], layers[2], stride=2)
        # 13, 13, 1024 -> 7, 7, 2048
        self.layer4 = self._make_layer(block, out_channels[3], layers[3], stride=2)
        self.layermany = self._make_layer(block, out_channels[3], layers[3], stride=2)
        self.layerfew = self._make_layer(block, out_channels[3], layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                       
    @property
    def out_channels(self):
        return [self.in_channels] + self._out_channels
        
    def _make_layer(
        self, 
        block: Type[Bottleneck],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
            
        layers = []

        layers.append(block(self.inplanes, planes, stride, downsample, dilation=previous_dilation, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes, dilation=self.dilation,
                    norm_layer=norm_layer)
                )
        return nn.Sequential(*layers)
    
    def _foward_impl(self, x): 
        x = self.conv1(x) 
        x = self.bn1(x)
        feat1 = self.relu(x) #64, 100, 100
        x = self.maxpool(feat1) # 64, 50, 50
        
        feat2 = self.layer1(x) # 256, 50, 50
        feat3 = self.layer2(feat2) # 512, 25, 25
        feat4 = self.layer3(feat3) # 1024, 13, 13
        feat5 = self.layer4(feat4) # 2048, 7, 7
        
        return [feat1, feat2, feat3, feat4, feat5]

    def forward(self, x):
        ls = self._foward_impl(x)
        return self._foward_impl(x)      
    
def getResNet50(pretrained=False, **kwargs):
    model = ResNet50(Bottleneck)
    if pretrained:
        weights = model_zoo.load_url(model_urls['resnet50'])
        # weights from channel(0) are copied for the new channel(dem)
        weight_dem = weights['conv1.weight'][:, 0:1]
        weights['conv1.weight'] = torch.cat((weights['conv1.weight'], weight_dem), dim = 1)

        model.load_state_dict(weights, strict=False)
    return model


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
    
class ACE_Res50_UNet(nn.Module):
    def __init__(self, num_classes, num_experts, train_LWS, train_MLP, pretrained = True):
        super(ACE_Res50_UNet, self).__init__()
        out_channels = [64, 128, 256, 512]
        self.resnet = getResNet50(pretrained = pretrained)
        in_channels = [192, 512, 1024, 3072]
        self.num_classes = num_classes
        self.num_experts = num_experts
        self.train_MLP = train_MLP
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
            if train_LWS:  
                self.SegHead_many = LWS(out_channels[0], self.num_classes)
                self.SegHead_few = LWS(out_channels[0], self.num_classes)            
            else:
                self.SegHead_many = nn.Conv2d(out_channels[0], self.num_classes, 1)
                self.SegHead_few = nn.Conv2d(out_channels[0], self.num_classes, 1)
        elif num_experts == 3:
            self.SegHead_many = nn.Conv2d(out_channels[0], self.num_classes, 1)
            self.SegHead_medium = nn.Conv2d(out_channels[0], self.num_classes, 1)
            self.SegHead_few = nn.Conv2d(out_channels[0], self.num_classes, 1)
        if self.train_MLP:
            self.MLP = MLP(num_inputs=20, hidden_layers=128)
        
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
            if self.train_MLP==True:
                # y_stack = torch.stack((y_many, y_few), -1)
                # MLP_output = self.MLP(y_stack)
                # MLP_output = torch.squeeze(MLP_output, -1)
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
        
# class SegmentationHead(nn.Sequential):
#     '''set up a segmentation head'''
#     def __init__(self, in_channels, out_channels, kernel_size = 3, upsampling = 1):
#         conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
#         upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
#         activation = nn.Identity()
#         super().__init__(conv2d, upsampling, activation)

class LWS(nn.Module):
    def __init__(self, num_features, num_classes):
        """Initialize the LWS layer.
        
        Args:
            num_features: number of features in the input tensor.
            num_classes: number of classes in the dataset.
        """
        super(LWS, self).__init__()
        self.conv2d = nn.Conv2d(num_features, num_classes, 1)
        self.scales = nn.Parameter(torch.ones(num_classes, 200, 200))
        for param_name, param in self.conv2d.named_parameters():
            param.requires_grad = False

    def forward(self, x):
        """Forward pass of the LWS layer."""
        x = self.conv2d(x)
        x *= self.scales
        return x

class MLP(nn.Module):
    def __init__(self, num_inputs, hidden_layers):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(num_inputs, hidden_layers),
                                 nn.ReLU(),
                                 nn.Linear(hidden_layers, 2),
                                )
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        x = self.mlp(x)
        x = self.softmax(x)
        return x

# class MLP(nn.Module):
#     def __init__(self, in_channels, out_channels,):
#         super(MLP, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
#         self.relu = nn.ReLU()
#         self.classifer = nn.Linear(out_channels, 2)
#         self.softmax = nn.Softmax(dim=-1)
#     def forward(self, x):
#         x = self.mlp(x)
#         # x = self.softmax(x)
#         return x

if __name__ == '__main__':
    model = ACE_Res50_UNet(10, 2, False, False)
    for k, v in model.named_parameters():
        # if k.startswith("SegHead"):
            print(k)
    