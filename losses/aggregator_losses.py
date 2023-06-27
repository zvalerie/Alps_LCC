import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from losses.ACE_losses import CELoss_2experts, CELoss_3experts


class AggregatorLoss(nn.Module):
    def __init__(self, args, ignore_index=0):
        super(AggregatorLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index= ignore_index)
        
        if args.experts == 2 :
            self.expertLoss = CELoss_2experts(args,ignore_index=ignore_index)
            
        elif args.experts == 3 : 
            self.expertLoss = CELoss_3experts(args, ignore_index=ignore_index)
    
    def forward(self, output, targets):
        
        expertLosses = self.expertLoss(output,targets)
        
        aggregationLoss = self.ce(output['aggregation'],targets)
        
        loss = torch.tensor(expertLosses).sum() + aggregationLoss
       
        return loss
        