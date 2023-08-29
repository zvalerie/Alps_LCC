import torch
from torch import nn
import torch.nn.functional as F

class CNN_merge(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers = 256):
        super(CNN_merge, self).__init__()
        self.cnn =  nn.Sequential(
                            nn.Conv2d(input_dim, hidden_layers, 3, padding=1, bias=False),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(2,1,padding=1),
                            nn.BatchNorm2d(hidden_layers),                            
                            nn.Conv2d(hidden_layers, output_dim, 1)
                        ) 
   
    def forward(self, x):
        x = torch.cat(x,dim=1)
        return self.cnn(x) 
    
class MLP_merge(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers = 512):
        
        super(MLP_merge, self).__init__()
        self.mlp = nn.Sequential(    
                    nn.Linear(input_dim, hidden_layers),
                    nn.ReLU(),
                    nn.Linear(hidden_layers, hidden_layers//2), 
                    nn.ReLU(),
                    nn.Linear( hidden_layers//2,output_dim),        
                    )

    def forward(self, x):
        x = torch.cat(x,dim=1)
        x = torch.moveaxis(x,1,-1)
        x = self.mlp(x)
        x = torch.moveaxis(x,-1,1)
        return x


class MLP_select(nn.Module):

    def __init__(self, num_experts, num_classes, hidden_layers = 512, return_expert_map=False):
        """
        Combine the prediction of several experts, by picking the best expert for each pixel.
        """
        super(MLP_select, self).__init__()
        self.num_experts = num_experts
        self.num_classes = num_classes
        self.return_expert_map = return_expert_map
        self.mlp = nn.Sequential(    
                    nn.Linear(num_experts* num_classes, hidden_layers),
                    nn.ReLU(),
                    nn.Linear(hidden_layers, hidden_layers//2), 
                    nn.ReLU(),
                    nn.Linear( hidden_layers//2, num_experts),        
                    )


    def forward(self, input):
        
        # Get prediction from each experts head : 
        size = input[0].size()  # 64x10x50x50
        head_logits = torch.cat(input,1)      
        head_logits = torch.movedim(head_logits,1,-1 )# 64x50x50 x30 
        head_logits = head_logits.flatten(end_dim=-2)#160000x30 
        
        # Select the expert with MLP : 
        exp_logits = self.mlp(head_logits) # [160000, 3])

        exp_logits = exp_logits.reshape(size[0],size[-2],size[-1],self.num_experts ).movedim(-1,1)
        

        return exp_logits

  
    
class CNN_select(nn.Module):
    def __init__(self, num_experts, num_classes, hidden_layers = 256, return_expert_map=False):
        """
        Combine the prediction of several experts, by picking the best expert for each pixel.
        """
        super(CNN_select, self).__init__()
        self.num_experts = num_experts
        self.num_classes = num_classes
        self.return_expert_map = return_expert_map
        self.cnn =  nn.Sequential(
                            nn.Conv2d(num_experts*num_classes, hidden_layers, 3, padding=1, bias=False),
                            nn.BatchNorm2d(hidden_layers),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(hidden_layers, num_experts, 1)
                        ) 

    def forward(self, input):
        
        
        # Get prediction from each experts head : 
        x = torch.stack(input,dim=2)
        
        # Select the expert with the CNN :
        input = torch.cat( input, dim=1 ) 
        exp_logits = self.cnn(input)
                
        return exp_logits


    
    
if __name__ == '__main__':
    
    x, y,z =  torch.rand([64,10,200,200]) ,torch.rand([64,10,200,200]) ,torch.rand([64,10,200,200]) 
   
    mlp = MLP_select(num_experts=3, num_classes=10,return_expert_map=True)
    out,map = mlp ([ x,y,z ])
    print(out.shape,map.shape )