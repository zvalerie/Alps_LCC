import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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


class WeightedCrossEntropyLoss(nn.Module):
    
    def __init__(self, ignore_index=0,args =None):
        
        super(WeightedCrossEntropyLoss, self).__init__()
        device = args.device if args is not None else "cuda"

        if args.ds =='TLM':
            inverse_freq_weights = torch.tensor(
                    [ 0.0,	3.5,	153.8,	7.9,	3.9,	
                     388.6,	3586.6,	3.4,	70.1,	118.4]   ).to(device)
        else : 
            # Flair with 19 classes 
            inverse_freq_weights = torch.tensor(
                    [ 0.,   11.9,    12.3,  7.1,    34.8,
                     21.8,  39.0,   6.5,    14.7,   31.5,
                     5.5,   9.0,    25.4,   3543.4, 585.3,     
                     592.2, 1688,   10472,  1289,               ]   
                     ).to(device)
            
            # Flair with 13 classes 
            #inverse_freq_weights = torch.tensor(
            #        [ 0.0,	11.9,	12.3,	7.1,	34.8,	
            #         21.8,	39.0,	 6.5,	14.7,	31.5,
            #         5.5,    9.0,   25.4 ]   
            #         ).to(device)

        
        self.ce = nn.CrossEntropyLoss(ignore_index= ignore_index,weight=inverse_freq_weights)

    def forward(self, output, targets):
    
        if isinstance(output,dict):
            return self.ce( output['out'], targets )
        
        else :
            return self.ce( output, targets )
        
        
class ClassBalancedLoss(nn.Module):
    
    def __init__(self, ignore_index=0,args =None):
        
        super(ClassBalancedLoss, self).__init__()
        device = args.device if args is not None else "cuda"
        # CBL computed with beta = 0.999


        if args.ds =='TLM': 
            CBL_weights = torch.tensor(
                    [0.0071,	0.0010,	0.0049,	0.0010,	
                    0.0010,	0.0117,	0.1046,	0.0010,	
                    0.0025,	0.0039,]  ).to(device)
        else : 
            # Flair with 19 classes 
            CBL_weights = torch.tensor(
                [    0,	0.0010,	0.0010,	0.0010,	0.0013,	
                0.0011,	0.0014,	0.0010,	0.0010,	0.0013,
                0.0010,	0.0010,	0.0012,	0.0721,	0.0123,	
                0.0125,	0.0346,	0.2120,	0.0266,]   
                     ).to(device)
                        
        
        self.ce = nn.CrossEntropyLoss(ignore_index= ignore_index,weight=CBL_weights)

    def forward(self, output, targets):
    
        if isinstance(output,dict):
            return self.ce( output['out'], targets )
        
        else :
            return self.ce( output, targets )




class CELoss_2experts(nn.Module):
    
    def __init__(self, args,ignore_index=0):
        
        super(CELoss_2experts, self).__init__()
        self.device = args.device
        if args.ds =='TLM':
            self.tail_index = torch.Tensor([2, 3, 4, 6, 7]).to(args.device)
            self.head_index =  torch.Tensor([1, 5, 8, 9]). to(args.device)
            self.tail_one_hot= torch.tensor(# 1 for tail index, 0 for head index
                [0, 0, 1, 1, 1, 0, 1, 1, 0, 0], 
                dtype=torch.float).to(self.device)
            
        else : #use FLAIR dataset with 19 classes
            self.tail_index = torch.Tensor([4,5,6,9,12,13,14,15,16,17,18]).to(args.device)
            self.head_index =  torch.Tensor([1,2,3,7,8,10,11]).to(args.device)
            self.tail_one_hot= torch.tensor( # 1 for tail index, 0 for head index
                [0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
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
                    
                                        
        return  head_loss , tail_loss
    
        


class CELoss_3experts(nn.Module):
    def __init__(self,  args, ignore_index=0):
        super(CELoss_3experts, self).__init__()
        self.device = args.device

        if args.ds =='TLM':
            self.tail_index = torch.tensor([3, 4]).to(args.device)
            self.body_index = torch.tensor([ 2, 3, 4, 6, 7 ]).to(args.device)
            self.head_index = torch.tensor([1, 5, 8, 9]).to(args.device)
            self.body_weight = torch.tensor([0, 0, 1, 1, 1, 0, 1, 1, 0, 0], 
                                            dtype=torch.float).to(args.device)
            self.tail_weight =    torch.tensor([0, 0, 0, 1, 1, 0, 0, 0, 0, 0], 
                                            dtype=torch.float).to(args.device)
            
        else : #use FLAIR dataset with 19 classes
            self.tail_index = torch.Tensor([13,14,15,16,17,18]).to(args.device)
            self.body_index = torch.Tensor([4,5,6,9,12]).to(args.device)
            self.head_index = torch.Tensor([1,2,3,7,8,10,11]).to(args.device)
            self.body_weight = torch.tensor( # 1 for tail and body index, 0 for head index
                [0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                dtype=torch.float).to(self.device)
            self.tail_weight = torch.tensor( # 1 for tail index, 0 for head and body index
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                dtype=torch.float).to(self.device)

        self.celoss= CrossEntropyLoss(ignore_index=ignore_index)
        self.body_masked_celoss = CrossEntropyLoss(weight=self.body_weight, ignore_index = ignore_index)
        self.tail_masked_celoss = CrossEntropyLoss(weight=self.tail_weight, ignore_index = ignore_index)
        self.use_L2_penalty = args.L2penalty
        if self.use_L2_penalty :
            self.complementary_loss = MSELoss(reduction='mean')


    def forward(self, output, targets):
        
        # Compute the classification loss for the head expert :
        head_loss = self.celoss(output['exp_0'], targets)  .unsqueeze(0)   
        
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
                body_loss +=  self.complementary_loss(mask.expand(size)*output['exp_1'], torch.zeros(size).to(self.device)) 
               
            
            
        # Compute the classification  loss from the tail expert, only if there are pixel from tail classes
        tail_loss = torch.Tensor([0.]).to(self.device)
        is_tail_pixel = torch.isin(targets, self.tail_index)        
        if is_tail_pixel.any():         
            tail_loss += self.tail_masked_celoss(output['exp_2'], targets)    
            
            # Compute complementary loss : 
            if self.use_L2_penalty:          
                # L2 penalty for predicting head and body classes from the tail expert 
                size = output['exp_2'].shape
                mask  = torch.logical_not(is_tail_pixel).long().unsqueeze(1)
                tail_loss += self.complementary_loss(mask.expand(size)*output['exp_2'], torch.zeros(size).to(self.device)) 
                    
                
                      
            
        return head_loss, body_loss, tail_loss
    
    
        


        
        
        

  