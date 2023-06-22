import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn.functional import cross_entropy

def MyCrossEntropyLoss(output, targets):
    
    return cross_entropy(output['out'],targets)

class ResCELoss_2experts(nn.Module):
    def __init__(self, args,device):
        super(ResCELoss_2experts, self).__init__()
        self.num_classes = 10
        self.few_index = [2, 3, 4, 6, 7]
        self.many_index = [1, 5, 8, 9]
        self.weight = torch.tensor(
            [0, 0, 1, 1, 1, 0, 1, 1, 0, 0], # 1 for few index, 0 for many index
            dtype=torch.float).to(device)
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
    
    
class ResCELoss_3experts(CrossEntropy2D):
    def __init__(self, many_index, medium_index, few_index, args):
        super(ResCELoss_3exp, self).__init__()
        self.num_classes = 10
        self.few_index = few_index
        self.medium_index = medium_index
        self.many_index = many_index
        self.medium_weight = torch.tensor([0, 0, 1, 1, 1, 0, 1, 1, 0, 0], dtype=torch.float).cuda()
        self.few_weight = torch.tensor([0, 0, 0, 1, 1, 0, 0, 0, 0, 0], dtype=torch.float).cuda()
        self.celoss= CrossEntropy2D()
        self.medium_masked_celoss = CrossEntropy2D(weight=self.medium_weight)
        self.few_masked_celoss = CrossEntropy2D(weight=self.few_weight)

    def forward(self, output, targets, **kwargs):
        
        [many_logits, medium_output_ori, many_logits], _= output
        few_loss = torch.tensor(0.0, requires_grad=True).to(many_logits.device)
        medium_loss = torch.tensor(0.0, requires_grad=True).to(many_logits.device)
        many_loss = torch.tensor(0.0, requires_grad=True).to(many_logits.device)
        targets_cpu = targets.cpu().numpy()
        
        ## among batch, return the idx that the mask contains few category
        medium_idx_cpu = [j for j in range(len(targets_cpu)) if any(np.isin(np.concatenate((self.few_index,self.medium_index)), targets_cpu[j]))] 
        few_idx_cpu = [j for j in range(len(targets_cpu)) if any(np.isin(self.few_index, targets_cpu[j]))] 
        
        if len(medium_idx_cpu):
        # compute weight celoss   
            medium_loss = self.medium_masked_celoss(medium_output_ori, targets)
            
        if len(few_idx_cpu):
        # compute weight celoss   
            few_loss = self.few_masked_celoss(many_logits, targets)
        
        many_loss = self.celoss(many_logits, targets)
        
        return [many_loss, medium_loss, few_loss], None