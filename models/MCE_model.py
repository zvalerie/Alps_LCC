import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.segmentation.deeplabv3 import ASPP
from torch.linalg import vector_norm
from torch.nn.functional import relu
from aggregator import CNN_merge, MLP_merge, MLP_select, CNN_select, MLP_moe

   
class Expert(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim = 256):
        super(Expert, self).__init__()
        self.cnn1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1, bias=False)
        self.bn1  = nn.BatchNorm2d(hidden_dim)
        self.cnn2 = nn.Conv2d(hidden_dim, output_dim, 1)
        _init_weight(self)
        
    def forward(self, x):
        x= self.cnn1(x)
        x=self.bn1(x)
        x= self.cnn2( relu (x,inplace=True))
        return x

   
    
class MCE(nn.Module):
    def __init__(self,  num_classes,  num_experts, use_lws=False, 
                 aggregation='mean'):
        
        super(MCE, self).__init__()
        
        assert num_experts  in [2,3 ], 'Num_experts must be 2 or 3 ! '
        self.num_experts = num_experts
        self.use_lws  = use_lws 
        self.aggregation = aggregation
        
        # Build model layers : 
        self.project = nn.Sequential( 
                nn.Conv2d(256, 48, 1, bias=False),
                nn.BatchNorm2d(48),
                nn.ReLU(inplace=True),
            )
        self.aspp = ASPP(2048, [12, 24, 36])

        self.expert_head = Expert(input_dim=304, output_dim=num_classes) 
        self.expert_tail = Expert(input_dim=304, output_dim=num_classes)          
        if num_experts == 3:
            self.expert_body = Expert(input_dim=304, output_dim=num_classes)   
        
        # Build classifier :
        if   aggregation == 'MLP_merge':
            self.classifier = MLP_merge(input_dim= num_experts * num_classes, output_dim= num_classes)    
            
        elif aggregation == 'CNN_merge':
            self.classifier = CNN_merge(input_dim= num_experts * num_classes, output_dim= num_classes)    
        
        elif aggregation == 'CNN_select':            
            self.classifier = CNN_select(num_experts, num_classes,)    
        
        elif aggregation == 'MLP_select':
            self.classifier = MLP_select(num_experts, num_classes,) 
        
        elif aggregation == 'MLP_moe':
            #raise NotImplementedError
            self.classifier = MLP_moe(num_experts=num_experts, num_classes=num_classes)
               
        else:
            self.classifier = nn.Sequential( nn.ReLU(inplace=True))    # Useless layer too avoid empty and bug
            
       
        


    def forward(self, feature):
        # DeepLabV3 backbone : 
        low_level_feature = self.project( feature['low_level'] )
        output_feature = self.aspp(feature['out'])
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False)
        output_feature = torch.cat( [ low_level_feature, output_feature ], dim=1 )       
        
        # Experts :
        y_tail = self.expert_tail(output_feature)
        y_head = self.expert_head(output_feature)        
        if self.num_experts == 3:   
            y_body = self.expert_body(output_feature)
        
        if self.use_lws :       
            f_tail= vector_norm(self.expert_head.cnn2.weight.flatten()) / vector_norm( self.expert_tail.cnn2.weight.flatten()) 
            y_tail= f_tail* y_tail
            if self.num_experts == 3: 
                f_body = vector_norm(self.expert_head.cnn2.weight.flatten()) / vector_norm( self.expert_body.cnn2.weight.flatten()) 
                y_body = f_body * y_body
        
        output_feature = [y_head, y_tail] if self.num_experts == 2 else [y_head, y_body, y_tail]
        
        # Aggregation methods : 
        aggregator_output = None
        if self.aggregation not in [ 'mean', 'max_pool']  :           
            aggregator_output = self.classifier(output_feature) 
           
        return output_feature, aggregator_output

        
def _init_weight(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)






if __name__ == '__main__':
    
    x = torch.rand([64,4,200,200])
    from models_utils import model_builder
    model = model_builder( 
                num_classes = 10, 
                pretrained_backbone = True, 
                num_experts = 3,
                use_lws = True,
                aggregation = 'mean',
                )
    output = model(x) 
    for key in output.keys():
        print('output i=',key, output[key].shape)
        
