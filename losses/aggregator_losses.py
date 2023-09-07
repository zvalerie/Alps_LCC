import torch
import torch.nn as nn
from losses.ACE_losses import CELoss_2experts, CELoss_3experts
from losses.selectExpertLoss import selectExpertLoss

class AggregatorLoss(nn.Module):
    def __init__(self, args, ignore_index=0):
        super(AggregatorLoss, self).__init__()
        self.aggregation = args.aggregation
        
        if args.reweighted_aggregation == 'CBL':# Weights from class balanced loss :  
            print('\t Uses CBL weights in the aggregator loss')
            weights = torch.Tensor ([0.0,   1e-3,      5e-3,    1e-2,            
                                     1e-1,  1e-3,     2e-3,     3e-3,       
                                     1e-3,  1e-3  ]).to(args.device)
            
            
        elif args.reweighted_aggregation == 'inverse_frequency':
            print('\t Uses inverse_frequency weights in the aggregator loss')
            weights= torch.tensor( # Weights from inverse frequency loss :
                    [ 0.0,	3.5,	153.8,	7.9,	3.9,	
                     388.6,	3586.6,	3.4,	70.1,	118.4]   ).to(args.device)
            
        else :
            weights = torch.ones(size = [10]).to(args.device)
            
        self.ce = nn.CrossEntropyLoss(ignore_index= ignore_index,weight= weights)
        self.selectExpertLoss = selectExpertLoss(args)
        self.device = args.device

        self.finetune_classifier_only  = args.finetune_classifier_only 
        if not self.finetune_classifier_only: # train full model
            if args.experts == 2 :
                self.expertLoss = CELoss_2experts(args,ignore_index=ignore_index)
                
            elif args.experts == 3 : 
                self.expertLoss = CELoss_3experts(args, ignore_index=ignore_index)
                
        
        
    def forward(self, output, targets):
        
        if 'select' in self.aggregation:
            
            loss = self.selectExpertLoss(output['aggregation'],targets)
            
        else :         
            
            loss = self.ce(output['aggregation'],targets)
            
        
        if  not self.finetune_classifier_only :
            
           exp_losses = torch.cat (self.expertLoss(output,targets)  ) .sum()
           loss = loss +exp_losses     
             
        return loss 
        