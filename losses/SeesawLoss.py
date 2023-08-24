import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

class SeesawLoss(nn.Module):
    """
    Seesaw Loss for Long-Tailed Instance Segmentation (CVPR 2021)
    arXiv: https://arxiv.org/abs/2008.10032
    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
             of softmax. Only False is supported.
        p (float, optional): The ``p`` in the mitigation factor.
             Defaults to 0.8.
        q (float, optional): The ``q`` in the compenstation factor.
             Defaults to 2.0.
        num_classes (int, optional): The number of classes.
             Default to 1203 for LVIS v1 dataset.
        eps (float, optional): The minimal value of divisor to smooth
             the computation of compensation factor
        reduction (str, optional): The method that reduces the loss to a
             scalar. Options are "none", "mean" and "sum".
    """

    def __init__(self,
                 num_classes,
                 reduction='mean',
                 ignore_index=0):
        super(SeesawLoss, self).__init__()
        self.p = 0.8
        self.q = 2.0
        self.num_classes = num_classes
        self.eps = 1e-2
        self.reduction = reduction
        self.register_buffer(
            'cum_samples',
            torch.zeros(self.num_classes, dtype=torch.float).cuda())
        self.cls_criterion = CrossEntropyLoss(ignore_index=ignore_index)
        
    def forward(self,
                cls_score,
                labels):
        """Forward function.
        Args:
            cls_score (torch.Tensor): The prediction with shape (N, C + 2).
            labels (torch.Tensor): The learning label of the prediction.
        Returns:
            torch.Tensor | Dict [str, torch.Tensor]:
                 if return_dict == False: The calculated loss |
                 if return_dict == True: The dict of calculated losses
                 for objectness and classes, respectively.
        """
        
        # accumulate the samples for each category
        if isinstance (cls_score,dict):
            cls_score = cls_score['out']
            
        unique_labels = labels.unique()
        self.cum_samples.cuda()
        for u_l in unique_labels:
            inds_ = labels == u_l.item()
            self.cum_samples[u_l.long()] += inds_.sum()
        labels = torch.flatten(labels) #[(B H W)]
        
        cls_score = torch.permute(cls_score, (0,2,3,1))
        cls_score = torch.flatten(cls_score, start_dim=0, end_dim=2)  
               
        cls_score = self.seesaw_logit(cls_score = cls_score, labels=labels, 
                                      cum_samples=self.cum_samples, num_classes= self.num_classes, 
                                      p = self.p, q = self.q, eps= self.eps
                                      )
        
        # calculate loss
        loss = self.cls_criterion(cls_score, labels)
        
        return loss

    def seesaw_logit(self,cls_score,
                    labels,
                    cum_samples,
                    num_classes,
                    p,
                    q,
                    eps):
        """Calculate the Seesaw CrossEntropy loss.
        Args:
            cls_score (torch.Tensor): The prediction with shape (N, C, H, W),
                C is the number of classes.
            labels (torch.Tensor): The learning label of the prediction.
            label_weights (torch.Tensor): Sample-wise loss weight.
            cum_samples (torch.Tensor): Cumulative samples for each category.
            num_classes (int): The number of classes.
            p (float): The ``p`` in the mitigation factor.
            q (float): The ``q`` in the compenstation factor.
            eps (float): The minimal value of divisor to smooth
                the computation of compensation factor
            reduction (str, optional): The method used to reduce the loss.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
        Returns:
            torch.Tensor: The calculated loss
        """
        assert cls_score.size(1) == num_classes
        assert len(cum_samples) == num_classes
        
        onehot_labels = F.one_hot(labels.long(), num_classes) # [(B H W) C]
        seesaw_weights = cls_score.new_ones(cls_score.size())  
        
        # mitigation factor
        if p > 0:
            sample_ratio_matrix = cum_samples[None, :].clamp(
                min=1) / cum_samples[:, None].clamp(min=1)
            index = (sample_ratio_matrix < 1.0).float()
            sample_weights = sample_ratio_matrix.pow(p) * index + (1 - index)
            mitigation_factor = sample_weights[labels.long(), :]
            seesaw_weights = seesaw_weights * mitigation_factor

        # compensation factor
        if q > 0:
            scores = F.softmax(cls_score.detach(), dim=1)
            self_scores = scores[
                torch.arange(0, len(scores)).to(scores.device).long(),
                labels.long()]
            score_matrix = scores / self_scores[:, None].clamp(min=eps)
            index = (score_matrix > 1.0).float()
            compensation_factor = score_matrix.pow(q) * index + (1 - index)
            seesaw_weights = seesaw_weights * compensation_factor

        cls_score = cls_score + (seesaw_weights.log() * (1 - onehot_labels))

        # loss = F.cross_entropy(cls_score, labels.long(), weight=None, reduction='none')
        
        return cls_score


if __name__ == '__main__':
    
    preds =torch.rand([16,10,50,50]).flatten(-1).cuda()
    target = torch.randint(high=10,size= [16,50,50]).cuda()
    print(preds.shape,target.shape)
    criterion = SeesawLoss(num_classes=10)
    loss = criterion(preds,target)
    print(loss)
