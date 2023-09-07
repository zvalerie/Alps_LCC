import torch
from torch import nn
import torch.nn.functional as F


def _init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)


class CNN_merge(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers = 256):
        super(CNN_merge, self).__init__()
        self.cnn =  nn.Sequential(
                            nn.Conv2d(input_dim, hidden_layers, 3, padding="same", bias=False),
                            nn.ReLU(inplace=True),
                            nn.BatchNorm2d(hidden_layers), 
                            nn.MaxPool2d(2,1,padding=1),                           
                            nn.Conv2d(hidden_layers, output_dim, 1)
                        ) 
        _init_weight(self.cnn)
        
    def forward(self, x):
        x = torch.cat(x,dim=1)
        return self.cnn(x) 
    
class MLP_merge(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers = 512):
        
        super(MLP_merge, self).__init__()
        self.mlp = nn.Sequential(    
                    nn.Linear(input_dim, hidden_layers),
                    nn.ReLU(),
                    nn.Linear( hidden_layers,output_dim),        
                    )
        _init_weight(self.mlp)

    def forward(self, x):
        x = torch.cat(x,dim=1)
        x = torch.moveaxis(x,1,-1)
        x = self.mlp(x)
        x = torch.moveaxis(x,-1,1)
        return x


class MLP_select(nn.Module):

    def __init__(self, num_experts, num_classes, hidden_layers = 16, return_expert_map=False):
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
        _init_weight(self.mlp)

    def forward(self, input):
        
        # Get prediction from each experts head :  
        input = [ F.normalize(x, p = 2, dim =1,) for x in input]
        experts_output = torch.cat(input,1)   
        experts_output = torch.movedim(experts_output,1,-1 )# BxHxW x nb_expert*nb_classes

        
        # Select the expert with MLP : 
        select_logits = self.mlp(experts_output) # [BxHxW, nb_expert])
        select_logits = select_logits.movedim(-1,1)
        

        return select_logits

  
    
class CNN_select(nn.Module):
    def __init__(self, num_experts, num_classes, hidden_layers = 16, return_expert_map=False):
        """
        Combine the prediction of several experts, by picking the best expert for each pixel.
        """
        super(CNN_select, self).__init__()
        self.num_experts = num_experts
        self.num_classes = num_classes
        self.return_expert_map = return_expert_map
        self.cnn =  nn.Sequential(
                            nn.Conv2d(num_experts*num_classes, hidden_layers, 3, padding='same'),
                            nn.BatchNorm2d(hidden_layers),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(hidden_layers, num_experts, 1)
                        ) 
        _init_weight(self.cnn)

    def forward(self, input):
        
        
        # Select the expert with the CNN :
        input = [ F.normalize(x, p = 2, dim =1,) for x in input]
        input = torch.cat( input, dim=1 ) 
        select_logits = self.cnn(input)
                
        return select_logits
    
    
    
class MLP_moe(nn.Module):
    
    def __init__(self, num_experts, num_classes, hidden_layers = 16, return_expert_map=False):
        """
        Combine the prediction of several experts, by giving a weight to each expert logits. (Mixture of expert)
        """
        super(MLP_moe, self).__init__()
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
        _init_weight(self.mlp)

    def forward(self, input):
        
        # Normalize and reshape the logits 
        input = [ F.normalize(x, p = 2, dim =1,) for x in input]
        experts_output = torch.cat(input,1)   
        experts_output = torch.movedim(experts_output,1,-1 )# BxHxW x nb_expert*nb_classes

        
        # Predict each expert weight for each pixel  with MLP : 
        mixture_logits = self.mlp(experts_output) # [BxHxW, nb_expert])
        mixture_logits = mixture_logits.movedim(-1,1) # [B x nb_expert x H x W])
        
        # Multiply  the logits for each pixel by their expert weight and sum them :
        logits = torch.stack(input,1)  # B xnb_ expert x nb_classes x H x W x nb_exp 
        logits = logits * mixture_logits.unsqueeze(2)
        logits = logits.sum(1)
        
        if self.return_expert_map :
            return logits, mixture_logits

        return logits


    
    
if __name__ == '__main__':
    
    x, y,z =  torch.rand([64,10,200,200]) ,torch.rand([64,10,200,200]) ,torch.rand([64,10,200,200]) 
   
    mlp = MLP_moe(num_experts=3, num_classes=10,return_expert_map=True)
    #mlp = CNN_merge(input_dim=30,output_dim=10)
    out, map = mlp ([ x,y,z ])
    print(out.shape )