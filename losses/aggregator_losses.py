import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from losses.ACE_losses import CELoss_2experts, CELoss_3experts


class AggregatorLoss(nn.Module):
    def __init__(self, args, ignore_index=0):
        super(AggregatorLoss, self).__init__()
        
                # Wegihts from class balanced loss :
        #'Background', "Bedrock", "Bedrockwith grass", "Large blocks", "Large blocks with grass", "Scree",  "Scree with grass","Water", "Forest" , "Glacier" ,
        weights = torch.Tensor ([0.0, 1e-3,      5e-3,               1e-2,             1e-1,                         1e-3 ,     2e-3,        3e-3,       1e-3,       1e-3  ]).to(args.device)
        print('uses weights in the aggregator loss')
        
        self.ce = nn.CrossEntropyLoss(ignore_index= ignore_index,weight= weights)

        

        if not args.finetune_classifier_only :
            if args.experts == 2 :
                self.expertLoss = CELoss_2experts(args,ignore_index=ignore_index)
                
            elif args.experts == 3 : 
                self.expertLoss = CELoss_3experts(args, ignore_index=ignore_index)
                
        self.finetune_classifier_only  = args.finetune_classifier_only 
        
    def forward(self, output, targets):
        
        if  not self.finetune_classifier_only :
            expertLosses = self.expertLoss(output,targets)
        else : 
            expertLosses=0.
        
        aggregationLoss = self.ce(output['aggregation'],targets)
        
        loss = torch.tensor(expertLosses).sum() + aggregationLoss
       
        return loss
        