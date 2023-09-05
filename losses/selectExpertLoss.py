import torch
import torch.nn as nn 


class selectExpertLoss(nn.Module):
    def __init__(self, args, ignore_index=0):
        super(selectExpertLoss, self).__init__()
        device = args.device
        assert args.experts ==3, 'not implemented for other than 3 experts'
        self.body_index = torch.tensor([2,  6, 7]).to(device)
        self.tail_index = torch.tensor([3, 4]).to(device)
        experts_weigths = torch.Tensor([1.04,34.,350.]).to(device)
        self.ce = nn.CrossEntropyLoss(weight=experts_weigths)
    
    def forward(self, output, targets):
        
        # Get GT for map of experts :     
        body_mask = torch.isin(targets,self.body_index) .float()
        tail_mask = torch.isin(targets, self.tail_index).float()
        expert_targets = (body_mask + tail_mask*2).long()
        
        # compute Crossentropy loss for network output :
        loss = self.ce(output, expert_targets )
        
        return loss
    
    