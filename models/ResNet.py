import math

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from typing import Type

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
    
    def __init__(self, inplanes, outplanes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=nn.BatchNorm2d):
        super(Bottleneck, self).__init__()
        width = int(outplanes * (base_width / 64.)) * groups
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
    
class ResNet50Encoder(nn.Module):
    def __init__(self, 
                 block = Bottleneck,
                 layers = [3, 4, 6, 3],
                 in_channels = 4,
                 out_channels = [64, 128, 256, 512],
                 groups=1, 
                 width_per_group=64,
                 norm_layer = nn.BatchNorm2d,
                 replace_stride_with_dilation=None
                ):
        super(ResNet50Encoder, self).__init__()
        
        self.in_channels = in_channels 
        self._out_channels = out_channels
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
            
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
        self.layer2 = self._make_layer(block, out_channels[1], layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        #25, 25, 512 -> 13, 13, 1024
        self.layer3 = self._make_layer(block, out_channels[2], layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        # 13, 13, 1024 -> 7, 7, 2048
        self.layer4 = self._make_layer(block, out_channels[3], layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

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
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, dilation=previous_dilation, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes, 
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
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
    
def resnet50(pretrained=False, **kwargs):
    model = ResNet50Encoder(block=Bottleneck, **kwargs)
    if pretrained:
        weights = model_zoo.load_url(model_urls['resnet50'])
        # weights from channel(0) are copied for the new channel(dem)
        weight_dem = weights['conv1.weight'][:, 0:1]
        weights['conv1.weight'] = torch.cat((weights['conv1.weight'], weight_dem), dim = 1)

        model.load_state_dict(weights, strict=False)
    return model

if __name__ == '__main__':
    
    x = torch.rand([32,4,200,200])
    model = resnet50(False)
    output = model(x) 
    print('input shape',x.shape)
    print('output shape', len(output),output[0].shape,output[1].shape,output[2].shape,output[3].shape,output[4].shape)