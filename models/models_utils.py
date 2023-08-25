from torch import nn
from torch.nn import functional as F
from ResNet import resnet50
from DeepLabV3Plus import IntermediateLayerGetter
from MCE_model import MCE 

def model_builder(num_classes, num_experts,  
                  pretrained_backbone=True, use_lws=False,
                  aggregation='mean'):

    backbone = resnet50(
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=[False, True, True])
    return_layers = {'layer4': 'out', 'layer1': 'low_level'}
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    
    classifier = MCE (  num_classes, 
                        num_experts, 
                        use_lws,
                        aggregation = aggregation,
                        )
    
    
    model = _MultiExpertModel(backbone, classifier, num_classes) 
       
    # give an informative name :
    name = "MCE"
    name += ' use_lws' if use_lws else ''
    name +=  ' with '+ aggregation  + ' aggregation '

    model.__class__.__name__ = name
    return model


class _MultiExpertModel(nn.Module):
    def __init__(self, backbone, classifier, num_classes):
        super(_MultiExpertModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.num_classes = num_classes
        
    def forward(self, x):
        output=dict()
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x, aggregator_output = self.classifier(features)
        
        if isinstance(x, list):
            for idx in range(len(x)):
                output[ 'exp_'+ str(idx)] = F.interpolate(x[idx], size=input_shape, mode='bilinear', align_corners=False) 
        else:
            output['out']= F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        
        if aggregator_output != None:
            if isinstance(aggregator_output,tuple) :
                x = F.interpolate(aggregator_output[0], size=input_shape, mode='bilinear', align_corners=False) 
                y = F.interpolate(aggregator_output[1].unsqueeze(1).float(), size=input_shape, mode='bilinear', align_corners=False) .long()
                output['aggregation'] =  (x,y.squeeze())      
            else : 
                output['aggregation']=  F.interpolate(aggregator_output, size=input_shape, mode='bilinear', align_corners=False) 
                
        
        return output
