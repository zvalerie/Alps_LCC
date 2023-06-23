import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn.functional import cross_entropy

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
    def __init__(self, args):
        super(CELoss_2experts, self).__init__()
        self.device = args.device
        
        self.few_index = [2, 3, 4, 6, 7]
        self.many_index = [1, 5, 8, 9]
        self.ls_index = [self.many_index, self.few_index]
        self.weight = torch.tensor(
            [0, 0, 1, 1, 1, 0, 1, 1, 0, 0], # 1 for few index, 0 for many index
            dtype=torch.float).to(self.device)
        self.celoss= CrossEntropyLoss
        self.masked_celoss = CrossEntropyLoss(weight=self.weight)
        
        

    def forward(self, output, targets):
        
        [many_logits, few_logits], _ = output
        
        # Compute a simple CE loss for the many expert :
        many_loss = self.celoss(many_logits, targets)

        # Compute the loss from the few expert only if there are pixel from few classes
        contain_few_pixel = torch.isin(targets, self.few_index).any()

        if contain_few_pixel: 
         
            few_loss = self.masked_celoss(few_logits, targets)

        
        
        return [many_loss, few_loss], None
    
    
class CELoss_3experts(nn.Module):
    def __init__(self,  args):
        super(CELoss_3experts, self).__init__()
        self.device = args.device
        self.few_index = [3, 4]
        self.medium_index = [2,3,4, 6]
        self.many_index = [1, 5, 7, 8, 9]
        self.medium_weight = torch.tensor([0, 0, 1, 1, 1, 0, 1, 0, 0, 0], dtype=torch.float).cuda()
        self.few_weight =    torch.tensor([0, 0, 0, 1, 1, 0, 0, 0, 0, 0], dtype=torch.float).cuda()
        self.celoss= MyCrossEntropyLoss()
        self.medium_masked_celoss = MyCrossEntropyLoss(weight=self.medium_weight)
        self.few_masked_celoss = MyCrossEntropyLoss(weight=self.few_weight)

    def forward(self, output, targets, **kwargs):
        
        [many_logits, medium_logits, many_logits], _= output
        
        # Compute a simple CE loss for the many expert :
        many_loss = self.celoss(many_logits, targets)
        
        # Compute the loss from the few expert only if there are pixel from few classes
        contain_few_pixel = torch.isin(targets, self.few_index).any()
        if contain_few_pixel:   
            few_loss = self.few_masked_celoss(many_logits, targets)
        
        # Compute the loss from the medium expert only if there are pixel from medium classes
        contain_medium_pixel = torch.isin(targets, self.medium_index ).any()
        if contain_medium_pixel : 
            medium_loss = self.medium_masked_celoss(medium_logits, targets)                   
        
        
        return [many_loss, medium_loss, few_loss], None