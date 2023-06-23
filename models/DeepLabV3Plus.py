import torch
from torchvision.models._utils import IntermediateLayerGetter
from models.models_utils import DeepLabHeadV3Plus, DeepLabV3
from ResNet import resnet50

def deeplabv3P_resnet(num_classes, output_stride=8, pretrained_backbone=True):

    if output_stride==8:
        replace_stride_with_dilation=[False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation=[False, False, True]
        aspp_dilate = [6, 12, 18]

    backbone = resnet50(
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation)
    
    inplanes = 2048
    low_level_planes = 256

    return_layers = {'layer4': 'out', 'layer1': 'low_level'}
    classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    model = DeepLabV3(backbone, classifier, num_classes)
    return model

if __name__ == '__main__':
    
    x = torch.rand([32,4,200,200])
    model = deeplabv3P_resnet(10)
    output =model(x) 
   
    print('input shape ',x.shape)
    print('output shape', output['out'].shape)