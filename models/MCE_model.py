import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.segmentation.deeplabv3 import ASPP
from torch.linalg import vector_norm
from torch.nn.functional import relu
from aggregator import CNN_merge, MLP_merge

   
class Expert(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim = 256):
        super(Expert, self).__init__()
        self.cnn1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1, bias=False)
        self.bn1  = nn.BatchNorm2d(hidden_dim)
        self.cnn2 = nn.Conv2d(hidden_dim, output_dim, 1)
   
    def forward(self, x):
        x= self.cnn1(x)
        x=self.bn1(x)
        x= self.cnn2( relu (x,inplace=True))
        return x

   
    
class MCE(nn.Module):
    def __init__(self,  num_classes,  num_experts, use_lws=False, 
                 use_CNN_aggregator=False, use_MLP_aggregator  = False):
        
        super(MCE, self).__init__()
        assert num_experts  in [2,3 ], 'Num_experts must be 2 or 3 ! '
        assert not ( use_MLP_aggregator and use_CNN_aggregator), 'only one of "use_MLP_aggregator" and "use_CNN_aggregator" can be True'
        self.num_experts = num_experts
        self.use_lws  = use_lws 
        
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
        
        if use_CNN_aggregator:
            self.classifier = CNN_merge(input_dim= num_experts * num_classes, output_dim= num_classes)    
            self.aggregation = 'cnn'
            
        elif use_MLP_aggregator :
            self.classifier = MLP_merge(input_dim= num_experts * num_classes, output_dim= num_classes)    
            self.aggregation = 'mlp'
        
        else:
            self.classifier = nn.Sequential( nn.ReLU(inplace=True))    # Useless layer too avoid empty and bug
            self.aggregation = None       
       
        


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
            
            
        if False : # New implementation seems to work less weel self.use_lws:       
            f_tail = vector_norm(self.expert_head.cnn2.weight,dim=1) / vector_norm(self.expert_tail.cnn2.weight,dim=1) 
            y_tail = f_tail * y_tail
            if self.num_experts == 3: 
                f_body = vector_norm(self.expert_head.cnn2.weight,dim=1) / vector_norm(self.expert_body.cnn2.weight,dim=1)
                y_body = f_body * y_body
        
        if self.use_lws :       
            f_tail= vector_norm(self.expert_head.cnn2.weight.flatten()) / vector_norm( self.expert_tail.cnn2.weight.flatten()) 
            y_tail= f_tail* y_tail
            if self.num_experts == 3: 
                f_body = vector_norm(self.expert_head.cnn2.weight.flatten()) / vector_norm( self.expert_body.cnn2.weight.flatten()) 
                y_body = f_body * y_body
            
        # Aggregation methods : 
        aggregator_output = None
        if self.aggregation is not None :

            if self.num_experts == 3:              
                aggregator_output = self.classifier(torch.cat([y_head, y_body, y_tail],dim=1) )
            elif self.num_experts == 2 :
                aggregator_output = self.classifier(torch.cat([y_head, y_tail],dim=1) )
                
            
        if self.num_experts == 3: 
            return [y_head, y_body, y_tail], aggregator_output
           
        return [y_head, y_tail], aggregator_output

        
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)






if __name__ == '__main__':
    
    x = torch.rand([8,4,100,200])
    from models_utils import model_builder
    model = model_builder( 
                num_classes = 10, 
                pretrained_backbone = True, 
                num_experts = 3,
                use_lws = True,
                use_MLP_aggregator = False,
                use_CNN_aggregator = True,
                )
    output = model(x) 
    for key in output.keys():
        print('output i=',key, output[key].shape)
        
