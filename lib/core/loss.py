import numpy as np
import torch
import torch.nn as nn

def _assert_no_grad(tensor):
    assert not tensor.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these tensors as not requiring gradients"
        
class FocalLoss(nn.Module):
    def __init__(self, ignore_index, alpha=0.5, gamma=2, weight=None, reduction = 'mean'):
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
        labels = torch.squeeze(labels,1)
        tb, th, tw = labels.size()

        assert(b == tb)

        # Handle inconsistent size between input and preds
        if resize_scores:
            if h != th or w != tw:  # upsample logits
                preds = nn.functional.interpolate(preds, size=(th, tw), mode="bilinear", align_corners=False)
        else:
            if h != th or w != tw:  # downsample labels
                labels = nn.functional.interpolate(labels.view(b, 1, th, tw).float(), size=(h, w), mode="nearest").view(b, h, w).long()
        
        logpt = -self.ce_fn(preds, labels.long())
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        return loss
    
    
class CrossEntropy2D(nn.Module):
    def __init__(self, reduction='mean', weight=None):
        """Initialize the module

        Args:
            ignore_index: specify which the label index to ignore.
            reduction (str): reduction method. See torch.nn.functional.cross_entropy for details.
            output_dir (str): output directory to save the checkpoint
            weight: weight for samples. See torch.nn.functional.cross_entropy for details.
        """
        super(CrossEntropy2D, self).__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, output, target, resize_scores=True):
        """Forward pass of the loss function

        Args:
            output (torch.nn.Tensor): output logits, i.e. network predictions w.o. softmax activation.
            target (torch.nn.Tensor): ground truth labels.
            resize_scores (bool): if set to True, when target and output have different widths or heights,
                                  upsample output bilinearly to match target resolution. Otherwise, downsample
                                  target using nearest neighbor to match input.
        Returns:
            loss (torch.nn.Tensor): loss between output and target.
        """
        _assert_no_grad(target)

        b, c, h, w = output.size()
        target = torch.squeeze(target,1)
        tb, th, tw = target.size()

        assert(b == tb)

        # Handle inconsistent size between input and target
        if resize_scores:
            if h != th or w != tw:  # upsample logits
                output = nn.functional.interpolate(output, size=(th, tw), mode="bilinear", align_corners=False)
        else:
            if h != th or w != tw:  # downsample labels
                target = nn.functional.interpolate(target.view(b, 1, th, tw).float(), size=(h, w), mode="nearest").view(b, h, w).long()

        loss = nn.functional.cross_entropy(
            output, target.long(), weight=self.weight, reduction=self.reduction
        )

        return loss