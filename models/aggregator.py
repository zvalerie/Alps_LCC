import torch
from torch import nn


class CNN_merge(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers = 256):
        super(CNN_merge, self).__init__()
        self.cnn =  nn.Sequential(
                            nn.Conv2d(input_dim, hidden_layers, 3, padding=1, bias=False),
                            nn.BatchNorm2d(hidden_layers),
                            nn.ReLU(inplace=True),
              #              nn.Dropout(p=0.2),
                            nn.Conv2d(hidden_layers, output_dim, 1)
                        ) 
   
    def forward(self, x):
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
        x = torch.moveaxis(x,1,-1)
        x = self.mlp(x)
        x = torch.moveaxis(x,-1,1)
        return x
