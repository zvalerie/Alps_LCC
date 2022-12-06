import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

def _assert_no_grad(tensor):
    assert not tensor.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these tensors as not requiring gradients"
        
def softmax_focal_loss_with_logits(
    output: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    reduction="mean",
    normalized=False,
    reduced_threshold: Optional[float] = None,
    ignore_index = 0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Softmax version of focal loss between target and output logits.
    See :class:`~pytorch_toolbelt.losses.FocalLoss` for details.
    Args:
        output: Tensor of shape [B, C, *] (Similar to nn.CrossEntropyLoss)
        target: Tensor of shape [B, *] (Similar to nn.CrossEntropyLoss)
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum' | 'batchwise_mean'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`.
            'batchwise_mean' computes mean loss per sample in batch. Default: 'mean'
        normalized (bool): Compute normalized focal loss (https://arxiv.org/pdf/1909.07829.pdf).
        reduced_threshold (float, optional): Compute reduced focal loss (https://arxiv.org/abs/1903.01347).
    """
    log_softmax = F.log_softmax(output, dim=1)

    loss = F.nll_loss(log_softmax, target, reduction="none", ignore_index=ignore_index)
    pt = torch.exp(-loss)

    # compute the loss
    if reduced_threshold is None:
        focal_term = (1.0 - pt).pow(gamma)
    else:
        focal_term = ((1.0 - pt) / reduced_threshold).pow(gamma)
        focal_term[pt < reduced_threshold] = 1

    loss = focal_term * loss

    if normalized:
        norm_factor = focal_term.sum().clamp_min(eps)
        loss = loss / norm_factor

    if reduction == "mean":
        loss = loss.mean()
    if reduction == "sum":
        loss = loss.sum()
    if reduction == "batchwise_mean":
        loss = loss.sum(0)

    return loss

class CrossEntropy2D(nn.Module):
    def __init__(self, ignore_index = 0, reduction='mean', weight=None):
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
        self.ignore_index = ignore_index

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
            output, target.long(), weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction
        )

        return loss

class ResCELoss(CrossEntropy2D):
    def __init__(self, many_index, few_index):
        super(ResCELoss, self).__init__()
        self.num_classes = 10
        self.few_index = few_index
        self.many_index = many_index
        self.weight = torch.tensor([0, 0, 1, 1, 1, 0, 1, 1, 0, 0], dtype=torch.float).cuda()
        self.celoss= CrossEntropy2D()
        self.weight_celoss = CrossEntropy2D(weight=self.weight)

    def forward(self, output, targets, **kwargs):
        
        [many_output_ori, few_output_ori]= output
        few_head_loss = torch.tensor(0.0, requires_grad=True).to(many_output_ori.device)
        many_head_loss = torch.tensor(0.0, requires_grad=True).to(many_output_ori.device)
        targets_cpu = targets.cpu().numpy()
        
        ## among batch, return the idx that the mask contains few category
        few_head_idx_cpu = [j for j in range(len(targets_cpu)) if any(np.isin(self.few_index, targets_cpu[j]))] 
        
        if len(few_head_idx_cpu):
        # compute weight celoss   
            claLoss = self.weight_celoss(few_output_ori, targets)
            comLoss = self._get_comLoss(few_output_ori, targets)
            few_head_loss = claLoss + comLoss
        
        many_head_loss = self.celoss(many_output_ori, targets)
        # few_head_loss = self.celoss(few_output_ori, targets)
        
        return many_head_loss + few_head_loss
    
    def _get_comLoss(self, output, targets):
        few_mask = (targets >= 2) & (targets <= 4) | (targets == 6) | (targets ==7)
        num_few_pixles = (few_mask == False).sum()
        # few_output = torch.masked_select(output, few_mask)
        few_mask = few_mask.expand(output.size())
        few_output = output * few_mask
        comLoss = torch.norm(few_output[:,self.many_index], p=2)/len(self.many_index)
        return comLoss