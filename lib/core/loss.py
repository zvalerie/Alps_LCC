import numpy as np
import torch
import torch.nn as nn

def _assert_no_grad(tensor):
    assert not tensor.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these tensors as not requiring gradients"
        
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, weight=None, ignore_index=0, reduction = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)

    def forward(self, preds, labels, resize_scores=True):
        '''Forward pass of the loss function'''
        _assert_no_grad(labels)
        
        b, c, h, w = preds.size()
        tb, th, tw = labels.size()

        assert(b == tb)

        # Handle inconsistent size between input and preds
        if resize_scores:
            if h != th or w != tw:  # upsample logits
                preds = nn.functional.interpolate(preds, size=(th, tw), mode="bilinear", align_corners=False)
        else:
            if h != th or w != tw:  # downsample labels
                labels = nn.functional.interpolate(labels.view(b, 1, th, tw).float(), size=(h, w), mode="nearest").view(b, h, w).long()
        
        logpt = -self.ce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        return loss
