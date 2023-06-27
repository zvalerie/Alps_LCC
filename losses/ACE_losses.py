import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss


class MyCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=0):
        super(MyCrossEntropyLoss, self).__init__()

        self.ce = nn.CrossEntropyLoss(ignore_index= ignore_index)

    def forward(self, output, targets):
        
        if isinstance(output,dict):
       
            return self.ce( output['out'], targets )
        else :
            return self.ce( output, targets )



class CELoss_2experts(nn.Module):
    def __init__(self, args,ignore_index=0):
        super(CELoss_2experts, self).__init__()
        self.device = args.device
        self.few_index = torch.Tensor([2, 3, 4, 6, 7]).to(args.device)
        self.many_index =  torch.Tensor([1, 5, 8, 9]). to(args.device)
        self.weight = torch.tensor(
            [0, 0, 1, 1, 1, 0, 1, 1, 0, 0], # 1 for few index, 0 for many index
            dtype=torch.float).to(args.device)
        self.celoss= CrossEntropyLoss(ignore_index=ignore_index)
        self.masked_celoss = CrossEntropyLoss( ignore_index=ignore_index, weight=self.weight)
        self.use_L2_penalty = args.L2penalty
        if self.use_L2_penalty :
            self.complementary_loss = MSELoss(reduction='mean')
               
                

    def forward(self, output, targets):       
        
        # Compute a simple CE loss for the many expert :
        many_loss = self.celoss(output['exp_0'], targets)

        # Compute the loss from the few expert only if there are pixel from few classes
        is_few_pixel = torch.isin(targets, self.few_index)
        
        if is_few_pixel.any(): 
            few_loss = self.masked_celoss(output['exp_1'], targets)
            
            if self.use_L2_penalty: # penalty for predicting frequent classes
                size = output['exp_1'].shape
                mask  = torch.logical_not(is_few_pixel).long().unsqueeze(1)
                few_loss += self.complementary_loss(mask.expand(size)*output['exp_1'], torch.zeros(size).to(self.device))             
                
        else :
            few_loss = torch.Tensor([0.]).to(self.device)
        
        
        return  many_loss , few_loss
    
        


class CELoss_3experts(nn.Module):
    def __init__(self,  args, ignore_index=0):
        super(CELoss_3experts, self).__init__()
        self.device = args.device
        self.few_index = torch.tensor([3, 4]).to(args.device)
        self.medium_index = torch.tensor([ 2, 3, 4, 6, 7 ]).to(args.device)
        self.many_index = torch.tensor([1, 5, 8, 9]).to(args.device)
        self.medium_weight = torch.tensor([0, 0, 1, 1, 1, 0, 1, 1, 0, 0], 
                                          dtype=torch.float).to(args.device)
        self.few_weight =    torch.tensor([0, 0, 0, 1, 1, 0, 0, 0, 0, 0], 
                                          dtype=torch.float).to(args.device)
        self.celoss= CrossEntropyLoss(ignore_index=0)
        self.medium_masked_celoss = CrossEntropyLoss(weight=self.medium_weight, ignore_index=0)
        self.few_masked_celoss = CrossEntropyLoss(weight=self.few_weight, ignore_index=0)
        self.use_L2_penalty = args.L2penalty
        if self.use_L2_penalty :
            self.complementary_loss = MSELoss(reduction='mean')
        self.use_LWS = args.lws

    def forward(self, output, targets, **kwargs):
        
        # Compute a simple CE loss for the many expert :
        many_loss = self.celoss(output['exp_0'], targets)     
        
        # Compute the loss from the medium expert only if there are pixel from medium classes
        is_medium_pixel = torch.isin(targets, self.medium_index )
        if is_medium_pixel.any() : 
            medium_loss = self.medium_masked_celoss(output['exp_1'], targets)
            
            if self.use_L2_penalty: # penalty for predicting frequent classes
                mask  = torch.logical_not(is_medium_pixel).long().unsqueeze(1)
                size = output['exp_1'].shape
                medium_loss += self.complementary_loss(mask.expand(size)*output['exp_1'], torch.zeros(size).to(self.device)) 
            
        else :
            medium_loss = torch.Tensor([0.]).to(self.device)

        # Compute the loss from the few expert only if there are pixel from few classes
        is_few_pixel = torch.isin(targets, self.few_index)
        if is_few_pixel.any():   
            
            few_loss = self.few_masked_celoss(output['exp_2'], targets)
            
            if  self.use_L2_penalty: # penalty for predicting frequent classes
                size = output['exp_2'].shape
                mask  = torch.logical_not(is_few_pixel).long().unsqueeze(1)
                few_loss += self.complementary_loss(mask.expand(size)*output['exp_2'], torch.zeros(size).to(self.device))        
            
        else :
            few_loss = torch.Tensor([0.]).to(self.device)
                      
            
        return many_loss, medium_loss, few_loss
    
    

  