#from coral_pytorch.dataset import levels_from_labelbatch
#from coral_pytorch.losses import coral_loss
import torch
import torch.nn.functional as F
from torch import nn


# Coral loss :
# https://raschka-research-group.github.io/coral-pytorch/api_subpackages/coral_pytorch.losses/#properties_1


class CoralLoss(nn.Module):
        
    def __init__(self, num_experts,device ='cpu'):
        super(CoralLoss, self).__init__()
        device = device
        self.body_index = torch.tensor([2,  6, 7]).to(device)
        self.tail_index = torch.tensor([3, 4]).to(device)
        experts_weigths = torch.Tensor([1.04,34.,350.]).to(device)
        self.num_experts = num_experts
       
        
    def forward(self,  logits, targets):
        # Get GT for map of experts :     
        body_mask = torch.isin(targets,self.body_index) .float()
        tail_mask = torch.isin(targets, self.tail_index).float()
        expert_targets = (1+body_mask + tail_mask*2).long() .flatten()
        print('EXPERT_TARGETS',expert_targets,expert_targets.shape)
        levels = levels_from_labelbatch(expert_targets,  num_classes= self.num_experts+1)
        print('LEVELS', levels,levels.shape)
        
        #### CORAL loss 
        loss = coral_loss(logits.view(-1,self.num_experts), levels)
        return loss
    
def coral_loss(logits, levels, importance_weights=None, reduction='mean'):
    """Computes the CORAL loss described in

    Cao, Mirjalili, and Raschka (2020)
    *Rank Consistent Ordinal Regression for Neural Networks
       with Application to Age Estimation*
    Pattern Recognition Letters, https://doi.org/10.1016/j.patrec.2020.11.008

    Parameters
    ----------
    logits : torch.tensor, shape(num_examples, num_classes-1)
        Outputs of the CORAL layer.

    levels : torch.tensor, shape(num_examples, num_classes-1)
        True labels represented as extended binary vectors
        (via `coral_pytorch.dataset.levels_from_labelbatch`).

    importance_weights : torch.tensor, shape=(num_classes-1,) (default=None)
        Optional weights for the different labels in levels.
        A tensor of ones, i.e.,
        `torch.ones(num_classes-1, dtype=torch.float32)`
        will result in uniform weights that have the same effect as None.

    reduction : str or None (default='mean')
        If 'mean' or 'sum', returns the averaged or summed loss value across
        all data points (rows) in logits. If None, returns a vector of
        shape (num_examples,)

    Returns
    ----------
        loss : torch.tensor
        A torch.tensor containing a single loss value
        (if `reduction='mean'` or '`sum'`)
        or a loss value for each data record (if `reduction=None`).

    Examples
    ----------
    >>> import torch
    >>> levels = torch.tensor(
    ...    [[1., 1., 0., 0.],
    ...     [1., 0., 0., 0.],
    ...    [1., 1., 1., 1.]])
    >>> logits = torch.tensor(
    ...    [[2.1, 1.8, -2.1, -1.8],
    ...     [1.9, -1., -1.5, -1.3],
    ...     [1.9, 1.8, 1.7, 1.6]])
    >>> coral_loss(logits, levels)
    tensor(0.6920)
    """

    if not logits.shape == levels.shape:
        raise ValueError(f"Please ensure that logits ({logits.shape})"
                         f" has the same shape as levels ({levels.shape}). ")

    term1 = (F.logsigmoid(logits)*levels
                      + (F.logsigmoid(logits) - logits)*(1-levels))

    if importance_weights is not None:
        term1 *= importance_weights

    val = (-torch.sum(term1, dim=1))

    if reduction == 'mean':
        loss = torch.mean(val)
    elif reduction == 'sum':
        loss = torch.sum(val)
    elif reduction is None:
        loss = val
    else:
        s = ('Invalid value for `reduction`. Should be "mean", '
             '"sum", or None. Got %s' % reduction)
        raise ValueError(s)

    return loss


def levels_from_labelbatch(labels, num_classes, dtype=torch.float32):
    """
    Converts a list of integer class label to extended binary label vectors
    Parameters
    ----------
    labels : list or 1D orch.tensor, shape=(num_labels,)
        A list or 1D torch.tensor with integer class labels
        to be converted into extended binary label vectors.
    num_classes : int
        The number of class clabels in the dataset. Assumes
        class labels start at 0. Determines the size of the
        output vector.
    dtype : torch data type (default=torch.float32)
        Data type of the torch output vector for the
        extended binary labels.
    Returns
    ----------
    levels : torch.tensor, shape=(num_labels, num_classes-1)
    Examples
    ----------
    >>> levels_from_labelbatch(labels=[2, 1, 4], num_classes=5)
    tensor([[1., 1., 0., 0.],
            [1., 0., 0., 0.],
            [1., 1., 1., 1.]])
    """
    levels = []
    for label in labels:
        levels_from_label = label_to_levels(
            label=label, num_classes=num_classes, dtype=dtype)
        levels.append(levels_from_label)

    levels = torch.stack(levels)
    return levels


def label_to_levels(label, num_classes, dtype=torch.float32):
    """Converts integer class label to extended binary label vector
    Parameters
    ----------
    label : int
        Class label to be converted into a extended
        binary vector. Should be smaller than num_classes-1.
    num_classes : int
        The number of class clabels in the dataset. Assumes
        class labels start at 0. Determines the size of the
        output vector.
    dtype : torch data type (default=torch.float32)
        Data type of the torch output vector for the
        extended binary labels.
    Returns
    ----------
    levels : torch.tensor, shape=(num_classes-1,)
        Extended binary label vector. Type is determined
        by the `dtype` parameter.
    Examples
    ----------
    >>> label_to_levels(0, num_classes=5)
    tensor([0., 0., 0., 0.])
    >>> label_to_levels(1, num_classes=5)
    tensor([1., 0., 0., 0.])
    >>> label_to_levels(3, num_classes=5)
    tensor([1., 1., 1., 0.])
    >>> label_to_levels(4, num_classes=5)
    tensor([1., 1., 1., 1.])
    """
    if not label <= num_classes-1:
        raise ValueError('Class label must be smaller or '
                         'equal to %d (num_classes-1). Got %d.'
                         % (num_classes-1, label))
    if isinstance(label, torch.Tensor):
        int_label = label.item()
    else:
        int_label = label

    levels = [1]*int_label + [0]*(num_classes - 1 - int_label)
    levels = torch.tensor(levels, dtype=dtype)
    return levels

if __name__ == '__main__':
    
    preds =torch.rand([1,3,2,5])
    targets = torch.randint(high=10,size= [1,2,5])
    print('PRED, TARGET', preds.shape,targets.shape)
    criterion = CoralLoss(num_experts=3)
    loss = criterion(preds,targets)
    print(loss)
