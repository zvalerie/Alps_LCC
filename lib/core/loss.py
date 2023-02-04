import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC
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
    def __init__(self, many_index, few_index, args):
        super(ResCELoss, self).__init__()
        self.num_classes = 10
        self.args = args
        self.few_index = few_index
        self.many_index = many_index
        self.weight = torch.tensor([0, 0, 1, 1, 1, 0, 1, 1, 0, 0], dtype=torch.float).cuda()
        # self.weight = torch.tensor([0, 0, 1, 1, 1, 0, 1, 0, 0, 0], dtype=torch.float).cuda()
        self.celoss= CrossEntropy2D()
        self.weight_celoss = CrossEntropy2D(weight=self.weight)

    def forward(self, output, targets, **kwargs):
        
        [many_output_ori, few_output_ori], MLP_output= output
        few_loss = torch.tensor(0.0, requires_grad=True).to(many_output_ori.device)
        many_loss = torch.tensor(0.0, requires_grad=True).to(many_output_ori.device)
        comLoss = torch.tensor(0.0, requires_grad=True).to(many_output_ori.device)
        targets_cpu = targets.cpu().numpy()
        
        ## among batch, return the idx that the mask contains few category
        few_idx_cpu = [j for j in range(len(targets_cpu)) if any(np.isin(self.few_index, targets_cpu[j]))] 
        
        if len(few_idx_cpu):
        # compute weight celoss   
            claLoss = self.weight_celoss(few_output_ori, targets)
            comLoss = self._get_comLoss(few_output_ori, targets)
            few_loss = claLoss + comLoss
            # few_loss = claLoss
        
        many_loss = self.celoss(many_output_ori, targets)
        return [many_loss, few_loss], comLoss
    
    def _get_comLoss(self, output, targets):
        few_mask = (targets >= 2) & (targets <= 4) | (targets == 6) | (targets == 7) #[16,1,200,200] 2 3 4 6 7
        num_few_pixles = (few_mask == True).sum()
        # few_output = torch.masked_select(output, few_mask)
        few_mask = few_mask.expand(output.size()) #[16, 10, 200 ,200]
        few_output = output * few_mask
        if self.args.loss == 0:
            comLoss = torch.norm(few_output[:,self.many_index], p=2)/len(self.many_index)
        elif self.args.loss == 1:
            comLoss = torch.norm(few_output[:,self.many_index], p=2)/num_few_pixles * self.args.comFactor
        elif self.args.loss == 2:
            comLoss = torch.norm(few_output[:,self.many_index], p=2)/len(self.many_index) * self.args.comFactor
        return comLoss
    
class ResCELoss_3exp(CrossEntropy2D):
    def __init__(self, many_index, medium_index, few_index, args):
        super(ResCELoss_3exp, self).__init__()
        self.num_classes = 10
        self.few_index = few_index
        self.medium_index = medium_index
        self.many_index = many_index
        self.args = args
        # self.medium_weight = torch.tensor([0, 0, 1, 1, 1, 0, 1, 1, 0, 0], dtype=torch.float).cuda()
        self.medium_weight = torch.tensor([0, 0, 1, 1, 1, 0, 1, 0, 0, 0], dtype=torch.float).cuda()
        # self.few_weight = torch.tensor([0, 0, 1, 1, 1, 0, 0, 0, 0, 0], dtype=torch.float).cuda()
        self.few_weight = torch.tensor([0, 0, 0, 1, 1, 0, 0, 0, 0, 0], dtype=torch.float).cuda()
        self.celoss= CrossEntropy2D()
        self.medium_weight_celoss = CrossEntropy2D(weight=self.medium_weight)
        self.few_weight_celoss = CrossEntropy2D(weight=self.few_weight)

    def forward(self, output, targets, **kwargs):
        
        [many_output_ori, medium_output_ori, few_output_ori], _= output
        few_loss = torch.tensor(0.0, requires_grad=True).to(many_output_ori.device)
        medium_loss = torch.tensor(0.0, requires_grad=True).to(many_output_ori.device)
        many_loss = torch.tensor(0.0, requires_grad=True).to(many_output_ori.device)
        medium_comLoss = torch.tensor(0.0, requires_grad=True).to(many_output_ori.device)
        targets_cpu = targets.cpu().numpy()
        
        ## among batch, return the idx that the mask contains few category
        medium_idx_cpu = [j for j in range(len(targets_cpu)) if any(np.isin(self.medium_index + self.few_index, targets_cpu[j]))] 
        few_idx_cpu = [j for j in range(len(targets_cpu)) if any(np.isin(self.few_index, targets_cpu[j]))] 
        
        if len(medium_idx_cpu):
        # compute weight celoss   
            medium_claLoss = self.medium_weight_celoss(medium_output_ori, targets)
            medium_comLoss = self._get_medium_comLoss(medium_output_ori, targets)
            medium_loss = medium_claLoss + medium_comLoss
            # few_head_loss = claLoss
            
        if len(few_idx_cpu):
        # compute weight celoss   
            few_claLoss = self.few_weight_celoss(few_output_ori, targets)
            few_comLoss = self._get_few_comLoss(few_output_ori, targets)
            few_loss = few_claLoss + few_comLoss
            # few_head_loss = claLoss
        
        many_loss = self.celoss(many_output_ori, targets)
        # few_head_loss = self.celoss(few_output_ori, targets)
        
        return [many_loss, medium_loss, few_loss], medium_comLoss
    
    def _get_few_comLoss(self, output, targets):
        few_mask = (targets >= 3) & (targets <= 4) #[16,1,200,200]
        num_few_pixles = (few_mask == True).sum()
        # few_output = torch.masked_select(output, few_mask)
        few_mask = few_mask.expand(output.size()) #[16, 10, 200 ,200]
        few_output = output * few_mask
        if self.args.loss == 0:
            comLoss = torch.norm(few_output[:,self.many_index + self.medium_index], p=2)/len(self.many_index + self.medium_index)
        elif self.args.loss == 1:
            comLoss = torch.norm(few_output[:,self.many_index + self.medium_index], p=2)/num_few_pixles * self.args.comFactor
        return comLoss
    
    def _get_medium_comLoss(self, output, targets):
        medium_mask = (targets >= 2) & (targets <= 4) | (targets == 6) #[16,1,200,200]
        num_medium_pixles = (medium_mask == True).sum()
        # few_output = torch.masked_select(output, few_mask)
        medium_mask = medium_mask.expand(output.size()) #[16, 10, 200 ,200]
        medium_output = output * medium_mask
        if self.args.loss == 0:
            comLoss = torch.norm(medium_output[:,self.many_index], p=2)/len(self.many_index)
        elif self.args.loss == 1:
            comLoss = torch.norm(medium_output[:,self.many_index], p=2)/num_medium_pixles * self.args.comFactor
        return comLoss
    
def seesaw_logit(cls_score,
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

    # labels = torch.flatten(labels) #[(B H W)]
    # cls_score = torch.flatten(cls_score.permute(0,2,3,1), start_dim=0, end_dim=2) #[(B H W) C]
    
    onehot_labels = F.one_hot(labels.long(), num_classes) # [(B H W) C]
    seesaw_weights = cls_score.new_ones(cls_score.size())  

    # if p > 0 and q > 0:
    #     # mitigation factor
    #     sample_ratio_matrix = cum_samples[None, :].clamp(
    #         min=1) / cum_samples[:, None].clamp(min=1)
    #     index = (sample_ratio_matrix < 1.0).float()
    #     sample_weights = sample_ratio_matrix.pow(p) * index + (1 - index)
    #     mitigation_factor = cls_score.new_ones(cls_score.size()) 
    #     onehot_labels = cls_score.new_ones(cls_score.size()) 
    #     # compensation factor
    #     scores = F.softmax(cls_score.detach(), dim=1)
    #     self_scores = labels.new_ones(labels.size()) 
        
    #     for i in range(labels.shape[0]) :
    #         for j in range(labels.shape[2]): 
    #             for k in range(labels.shape[3]):
    #                 label = labels[i, 0, j, k].long()
    #                 mitigation_factor[i, :, j, k] = sample_weights[label, :]
    #                 # compensation factor
    #                 self_scores[i, 0, j, k] = scores[i, label, j, k]
    #                 onehot_labels[i, label, j , k ] = 0
        
    #     score_matrix = scores / self_scores.clamp(min=eps)
    #     # self_scores = scores[
    #     #     torch.arange(0, len(scores)).to(scores.device).long(),
    #     #     labels.long()]
    #     # score_matrix = scores / self_scores[:, None].clamp(min=eps)
    #     index = (score_matrix > 1.0).float()
    #     compensation_factor = score_matrix.pow(q) * index + (1 - index)
        
    #     seesaw_weights = mitigation_factor * compensation_factor
    #     print('seesaw_weight done')
    # cls_score = cls_score + (seesaw_weights.log() * (1 - onehot_labels))
    
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

    def get_one_hot(label, num_classes):
        size = list(labels.size())
        labels= labels.view(-1)
        ones = torch.sparse.torch.eye(10)
        ones = ones.index_select(0, labels)
        ones = ones.permute(1,0).unsqueeze(dim=2)


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
        loss_weight (float, optional): The weight of the loss. Defaults to 1.0
        return_dict (bool, optional): Whether return the losses as a dict.
             Default to True.
    """

    def __init__(self,
                 p=0.8,
                 q=2.0,
                 num_classes=10,
                 eps=1e-2,
                 reduction='mean',
                 ignore_index=0):
        super(SeesawLoss, self).__init__()
        self.p = p
        self.q = q
        self.num_classes = num_classes
        self.eps = eps
        self.reduction = reduction
        self.register_buffer(
            'cum_samples',
            torch.zeros(self.num_classes, dtype=torch.float))
        self.cls_criterion = CrossEntropyLoss(ignore_index=ignore_index)
        
    def forward(self,
                cls_score,
                labels):
        """Forward function.
        Args:
            cls_score (torch.Tensor): The prediction with shape (N, C + 2).
            labels (torch.Tensor): The learning label of the prediction.
            label_weights (torch.Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor that is used to average
                 the loss. Defaults to None.
            reduction (str, optional): The method used to reduce the loss.
                 Options are "none", "mean" and "sum".
        Returns:
            torch.Tensor | Dict [str, torch.Tensor]:
                 if return_dict == False: The calculated loss |
                 if return_dict == True: The dict of calculated losses
                 for objectness and classes, respectively.
        """
        
        # accumulate the samples for each category
        unique_labels = labels.unique()
        self.cum_samples.cuda()
        for u_l in unique_labels:
            inds_ = labels == u_l.item()
            self.cum_samples[u_l.long()] += inds_.sum()
        labels = torch.flatten(labels) #[(B H W)]
        cls_score = torch.permute(cls_score, (0,2,3,1))
        cls_score = torch.flatten(cls_score, start_dim=0, end_dim=2) 
        
        cls_score = seesaw_logit(cls_score, labels, self.cum_samples, self.num_classes, self.p, self.q, self.eps)
        # calculate loss
        loss = self.cls_criterion(cls_score, labels)
        return loss
    
    
class PPC(nn.Module, ABC):
    def __init__(self, ignore_index=0):
        super(PPC, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, contrast_logits, contrast_target):
        loss_ppc = F.cross_entropy(contrast_logits, contrast_target.long(), ignore_index=self.ignore_index)

        return loss_ppc


class PPD(nn.Module, ABC):
    def __init__(self, ignore_index=0):
        super(PPD, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, contrast_logits, contrast_target):
        contrast_logits = contrast_logits[contrast_target != self.ignore_index, :]
        contrast_target = contrast_target[contrast_target != self.ignore_index]

        logits = torch.gather(contrast_logits, 1, contrast_target[:, None].long())
        loss_ppd = (1 - logits).pow(2).mean()

        return loss_ppd

class PixelPrototypeCELoss(nn.Module, ABC):
    def __init__(self, loss_ppc_weight=0.01, loss_ppd_weight=0.01):
        super(PixelPrototypeCELoss, self).__init__()
        ignore_index = 0

        self.loss_ppc_weight = loss_ppc_weight
        self.loss_ppd_weight = loss_ppd_weight
        self.seg_criterion = CrossEntropy2D()
        # self.seg_criterion = SeesawLoss()

        self.ppc_criterion = PPC()
        self.ppd_criterion = PPD()

    def forward(self, preds, target):
        # h, w = target.size(1), target.size(2)
        h, w = target.size(2), target.size(3)
        if isinstance(preds, dict):
            assert "seg" in preds
            assert "logits" in preds
            assert "target" in preds

            seg = preds['seg']
            contrast_logits = preds['logits']
            contrast_target = preds['target']
            loss_ppc = self.ppc_criterion(contrast_logits, contrast_target)
            loss_ppd = self.ppd_criterion(contrast_logits, contrast_target)

            pred = F.interpolate(input=seg, size=(h, w), mode='bilinear', align_corners=True)
            loss = self.seg_criterion(pred, target)
            return loss + self.loss_ppc_weight * loss_ppc + self.loss_ppd_weight * loss_ppd

        seg = preds
        pred = F.interpolate(input=seg, size=(h, w), mode='bilinear', align_corners=True)
        loss = self.seg_criterion(pred, target)
        return loss
    

class CrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index = 0, reduction='mean', weight=None):
        """Initialize the module

        Args:
            ignore_index: specify which the label index to ignore.
            reduction (str): reduction method. See torch.nn.functional.cross_entropy for details.
            output_dir (str): output directory to save the checkpoint
            weight: weight for samples. See torch.nn.functional.cross_entropy for details.
        """
        super(CrossEntropyLoss, self).__init__()
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

        n, c = output.size()
        tn = target.size()
        
        assert(n == tn[0])

        loss = nn.functional.cross_entropy(
            output, target.long(), weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction
        )

        return loss


if __name__ == '__main__':
    torch.manual_seed(1)
    labels = torch.randint(0,10,[2,1,4,4], dtype = torch.int32)
    cls_score = torch.randint(1,10,[2,10,4,4], dtype = torch.float32)
    criteron = SeesawLoss()
    loss = criteron(cls_score, labels)