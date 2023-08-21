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
        self.tail_index = torch.Tensor([2, 3, 4, 6, 7]).to(args.device)
        self.head_index =  torch.Tensor([1, 5, 8, 9]). to(args.device)
        self.tail_one_hot= torch.tensor(
            [0, 0, 1, 1, 1, 0, 1, 1, 0, 0], # 1 for tail index, 0 for head index
            dtype=torch.float).to(self.device)
        self.celoss= CrossEntropyLoss(ignore_index=ignore_index)
        self.masked_celoss = CrossEntropyLoss( ignore_index=ignore_index, weight=self.tail_one_hot)
        self.use_L2_penalty = args.L2penalty
        if self.use_L2_penalty :
            self.complementary_loss = MSELoss(reduction='mean')
              
                

    def forward(self, output, targets):       
        
        # Compute the classification loss for the head expert :
        head_loss = self.celoss(output['exp_0'], targets)
        
        # Compute the classification loss from the tail expert only if there are pixel from tail classes
        tail_loss = torch.Tensor([0.]).to(self.device)
        is_tail_pixel = torch.isin(targets, self.tail_index)        
        if is_tail_pixel.any(): 
            tail_loss += self.masked_celoss(output['exp_1'], targets)
        
            # Compute complementary loss if the tail head is in use : 
            # L2 penalty for predicting frequent classes from the tail expert
            if self.use_L2_penalty: 
                    size = output['exp_1'].shape
                    mask  = torch.logical_not(is_tail_pixel).long().unsqueeze(1)
                    tail_loss +=  self.complementary_loss(mask.expand(size)*output['exp_1'], torch.zeros(size).to(self.device)) 
                    #tail_loss += torch.norm ( mask.expand(size)*output['exp_1'] )
                                        
        return  head_loss , tail_loss
    
        


class CELoss_3experts(nn.Module):
    def __init__(self,  args, ignore_index=0):
        super(CELoss_3experts, self).__init__()
        self.device = args.device
        self.tail_index = torch.tensor([3, 4]).to(args.device)
        self.body_index = torch.tensor([ 2, 3, 4, 6, 7 ]).to(args.device)
        self.head_index = torch.tensor([1, 5, 8, 9]).to(args.device)
        self.body_weight = torch.tensor([0, 0, 1, 1, 1, 0, 1, 1, 0, 0], 
                                          dtype=torch.float).to(args.device)
        self.tail_weight =    torch.tensor([0, 0, 0, 1, 1, 0, 0, 0, 0, 0], 
                                          dtype=torch.float).to(args.device)
        self.celoss= CrossEntropyLoss(ignore_index=ignore_index)
        self.body_masked_celoss = CrossEntropyLoss(weight=self.body_weight, ignore_index = ignore_index)
        self.tail_masked_celoss = CrossEntropyLoss(weight=self.tail_weight, ignore_index = ignore_index)
        self.use_L2_penalty = args.L2penalty
        if self.use_L2_penalty :
            self.complementary_loss = MSELoss(reduction='mean')


    def forward(self, output, targets):
        
        # Compute the classification loss for the head expert :
        head_loss = self.celoss(output['exp_0'], targets)     
        
        # Compute the classification loss from the body expert, only if there are pixel from body classes
        body_loss = torch.Tensor([0.]).to(self.device)
        is_body_pixel = torch.isin(targets, self.body_index )        
        if is_body_pixel.any() : 
            body_loss += self.body_masked_celoss(output['exp_1'], targets)
            
            # Compute complementary loss : 
            if self.use_L2_penalty:
                # L2 penalty for predicting head  classes from the body expert 
                size = output['exp_1'].shape   
                mask  = torch.logical_not(is_body_pixel).long().unsqueeze(1)
                body_loss += torch.norm ( mask.expand(size)*output['exp_1'] )   
            
            
        # Compute the classification  loss from the tail expert, only if there are pixel from tail classes
        tail_loss = torch.Tensor([0.]).to(self.device)
        is_tail_pixel = torch.isin(targets, self.tail_index)        
        if is_tail_pixel.any():         
            tail_loss += self.tail_masked_celoss(output['exp_2'], targets)    
            
        # Compute complementary loss : 
        if self.use_L2_penalty:
            # L2 penalty for predicting head  classes from the body expert 
            size = output['exp_1'].shape   
            mask  = torch.logical_not(is_body_pixel).long().unsqueeze(1)
            body_loss += self.complementary_loss(mask.expand(size)*output['exp_1'], torch.zeros(size).to(self.device)) 
            #body_loss += torch.norm ( mask.expand(size)*output['exp_1'] )  
            
            # Compute complementary loss : 
            if self.use_L2_penalty:          
                # L2 penalty for predicting head and body classes from the tail expert 
                size = output['exp_2'].shape
                mask  = torch.logical_not(is_tail_pixel).long().unsqueeze(1)
                tail_loss += self.complementary_loss(mask.expand(size)*output['exp_2'], torch.zeros(size).to(self.device)) 
              #  tail_loss += torch.norm ( mask.expand(size)*output['exp_2'] )       
                
                      
            
        return head_loss, body_loss, tail_loss
    
    
        


        
        
        

  