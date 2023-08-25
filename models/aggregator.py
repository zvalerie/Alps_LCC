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
    def __init__(self, input_dim, output_dim, hidden_layers = 256):
        
        super(MLP_merge, self).__init__()
        self.mlp = nn.Sequential(    
                    nn.Linear(input_dim, hidden_layers),
                    nn.ReLU(),
                    nn.Linear(hidden_layers, output_dim),              
                    )

    def forward(self, x):
        x = torch.cat(x,dim=1)
        x = torch.moveaxis(x,1,-1)
        x = self.mlp(x)
        x = torch.moveaxis(x,-1,1)
        return x


class MLP_select(nn.Module):

    def __init__(self, num_experts, num_classes, hidden_layers = 256, return_expert_map=False):
        """
        Combine the prediction of several experts, by picking the best expert for each pixel.
        """
        super(MLP_select, self).__init__()
        self.num_experts = num_experts
        self.num_classes = num_classes
        self.return_expert_map = return_expert_map
        self.mlp = nn.Sequential(
            nn.Linear(in_features= num_experts*num_classes,out_features=hidden_layers),
            nn.ReLU(),
            nn.Linear(hidden_layers, num_experts)            
        )


    def forward(self, input):
        
        # Get prediction from each experts head : 
        size = input[0].size()
        head_logits = torch.stack(input,0)
        head_logits = torch.movedim(head_logits,2,1 ).reshape(self.num_experts,self.num_classes, -1)

        
        # Select the expert with MLP : 
        x = head_logits.reshape(self.num_classes*self.num_experts,-1).t()
        exp_logits = self.mlp(x)
        exp_selected = torch.argmax( F.softmax( exp_logits ,dim = 1),dim=-1)
        exp_selected_one_hot = F.one_hot(exp_selected).t()  # shape is nb_pixel, nb experts     
            

        # Get the prediction from the selected expert for each pixel :        
        output = ( exp_selected_one_hot.unsqueeze(1) *head_logits ).sum(dim=0)
        output =  output .reshape ([self.num_classes, size[0],size[2],size[3]])
        output =  torch.movedim(output,1,0 )
        

        return output, exp_selected.reshape ([size[0],size[2],size[3]])


    
    
    
    
    
    
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
        size = input[0].size()
        x = torch.stack(input,dim=2)
        
        # Select the expert with the CNN :
        input = torch.cat( input, dim=1 ) 
        exp_logits = self.cnn(input)
        exp_selected = torch.argmax( F.softmax( exp_logits ,dim = 1),dim=1)
        exp_selected_one_hot = F.one_hot(exp_selected).movedim(-1,1)  # shape is nb_pixel, nb experts     
            

        # Get the prediction from the selected expert for each pixel : 
        exp_selected_one_hot =     exp_selected_one_hot.reshape(size[0],-1)  .unsqueeze(1) 
        x = x.reshape(size[0],self.num_classes, -1)
        output = ( exp_selected_one_hot *x )
        output = output.reshape ( size[0],self.num_classes,self.num_experts,  size[2],size[3])
        output =  output.sum(2)
                
        return output, exp_selected


    
    
if __name__ == '__main__':
    
    x, y,z =  torch.rand([64,10,200,200]) ,torch.rand([64,10,200,200]) ,torch.rand([64,10,200,200]) 
   
    mlp = MLP_select(num_experts=3, num_classes=10,return_expert_map=True)
    out,map = mlp ([ x,y,z ])
    print(out.shape,map.shape )