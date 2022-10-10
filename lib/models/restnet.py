from tokenize import group
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from typing import Type, List, Optional

# urls for pretrained models
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
                 dilation=1, norm_layer=nn.BatchNorm2d) -> None:
        super().__init__()
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
    
class ResNetEncoder(nn.Module):
    def __init__(self, 
                 block = Bottleneck,
                 layers = [3, 4, 6, 3],
                 in_channels = 4,
                 out_channels = [64, 128, 256, 512],
                 norm_layer = nn.BatchNorm2d
                ):
        super(ResNetEncoder, self).__init__
        
        self.in_channels = in_channels 
        self._out_channels = out_channels
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        
        # input layer
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        #conv layers
        self.layer1 = self._make_layer(block, out_channels[0], layers[0])
        self.layer2 = self._make_layer(block, out_channels[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, out_channels[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, out_channels[3], layers[3], stride=2)

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
                    planes, stride, downsample, dilation=self.dilation,
                    norm_layer=norm_layer)
                )
        return nn.Sequential(*layers)
    
    def _foward_impl(self, x): 
        features = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        features.append(x)
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        features.append(x)
        x = self.layer4(x)
        features.append(x)
        return features
        