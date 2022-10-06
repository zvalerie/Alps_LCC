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
    
    def __init__(self, inplanes, planes, stride=1, downsample=None, base_width=64, 
                 dilation=1, norm_layer=None) -> None:
        super(Bottleneck).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0))
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        
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
    
class ResNetEncoder():
    def __init__(self, 
                 block: Type[Bottleneck],
                 layers: List[int] = [3, 4, 6, 3],
                 in_channels: int = 3,
                 out_channels: List[int] = [256, 512, 1024, 2048],
                 aux_in_channels: Optional[int] = None,
                 aux_in_position: Optional[int] = None
                ):
        super(ResNetEncoder, self).__init__
        
        self.in_channels = in_channels 
        self.out_channels = out_channels
        self.aux_in_position = aux_in_position
        self.norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.dilation = 1
        
        # input layer
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self.norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        #conv layers
        self.layer1 = self.
        
        
    def _make_layer(
        self, 
        block: Type[Bottleneck],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self.norm_layer
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
                    planes, stride, downsample, dilation=previous_dilation,
                    norm_layer=norm_layer)
                )
        return nn.Sequential(*layers)