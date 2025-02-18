import torch
import numpy as np
import torch.nn.functional as F
from torch.nn.functional import softmax

import os


def load_best_model_weights (model,args):
    best_model_path = os.path.join( args.out_dir, args.name,'current_best.pt')
    last_model_path = os.path.join( args.out_dir, args.name,'last_model.pt')
    
    if os.path.isfile (best_model_path) :
        checkpoint = torch.load(best_model_path)
        print('Weights loaded from best model path:', best_model_path)
        
    elif os.path.isfile(last_model_path):
        checkpoint = torch.load(last_model_path)   
        print('Using last model weights, best weights not found')     
    
    else :
        raise NameError ('Best or last model weights are not found in folder', args.out_dir , args.name )
    
    best_weights = checkpoint['state_dict']
    model =  model.load_state_dict (best_weights)


class AverageMeter(object):
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
def get_predictions_from_logits(output,args):
    
    
    if args.zero_nontarget_expert : 
        device = output['exp_1'].device
        output['exp_1'] =   torch.tensor([0., 0., 1., 1., 1., 0., 1., 1., 0., 0.])[:,None,None].to(device) * output['exp_1']
        
        if args.experts ==3  :
           output['exp_2'] = torch.tensor([0., 0., 0., 1., 1., 0., 0., 0., 0., 0.])[:,None,None].to(device) * output['exp_2']
        
    
    if  isinstance(output, torch.TensorType):
       logits = output
       
    elif 'merge' in args.aggregation:
        logits =   output['aggregation'] 
        
    elif 'moe' in args.aggregation:
        logits =   output['aggregation'] 
       
       
    elif 'select' in  args.aggregation :
        
        logits =   output['aggregation']

        # Get the selected expert from aggregation method :
        exp_selected = torch.argmax( softmax( logits ,dim = 1),dim=1)
        exp_selected_one_hot = F.one_hot(exp_selected,num_classes = args.experts)
        exp_selected_one_hot = exp_selected_one_hot.movedim(-1,1).unsqueeze(1)  # shape is nb_pixel, nb experts 
        
        # aggregatate and select the class logits :
        experts_logits = [ output['exp_0'],output['exp_1'], output['exp_2']] if args.experts ==3 else \
                         [ output['exp_0'],output['exp_1']]
        
        x = torch.stack(experts_logits,dim=2)
        
        logits = ( exp_selected_one_hot *x ).sum(1)
        
     
    elif 'out' in output.keys():
        logits = output['out']    
    
    elif  args.aggregation == 'mean' :
        # Aggregation is simple average : the output logits of class c is the average among the ouputs 
        # from set of experts that trained with class c,  then softmax is applied. 
        
        device = output['exp_1'].device
        sum_logits = ( output['exp_0'] + output['exp_1'] ) if args.experts ==2  \
            else  ( output['exp_0'] + output['exp_1'] + output['exp_2'] )
        # Divide the sum of logits by the number of experts used to predict them :
        quotient = [1,1,2,2,2,1,2,2,1,1] if args.experts ==2  \
            else   [1,1,2,3,3,1,2,2,1,1]
        quotient = torch.tensor(quotient).unsqueeze(-1).unsqueeze(-1).to(device)
        logits = sum_logits / quotient
            
                
        
    elif  args.aggregation == 'max_pool' :         
        group = torch.stack( output['exp_0'] , output['exp_1'] ) if args.experts ==2  \
            else  torch.stack( [output['exp_0'], output['exp_1'], output['exp_2']])
        logits = group.max(dim=0)[0]
         
        
    preds = torch.argmax(logits.detach().cpu(),axis=1)    
    return preds    


class MetricLogger(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.many_idx = [1, 5, 8, 9]
        self.medium_idx = [2, 6, 7]
        self.few_idx = [3, 4]
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask],
            minlength=n_class ** 2,
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(
                lt.flatten(), lp.flatten(), self.n_classes
            )

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        with np.errstate(divide='ignore', invalid='ignore'):
            acc_cls = np.diag(hist) / hist.sum(axis=1)
        mean_cls = np.nanmean(acc_cls)
        with np.errstate(divide='ignore', invalid='ignore'):
            iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        
        return mean_cls, mean_iu, acc_cls, acc

    def get_acc_cat(self):
        """Returns accuracy of different categories."""
        hist = self.confusion_matrix
        many_acc = np.diag(hist)[self.many_idx].sum() / hist.sum(axis=1)[self.many_idx].sum()
        medium_acc = np.diag(hist)[self.medium_idx].sum() / hist.sum(axis=1)[self.medium_idx].sum()
        few_acc = np.diag(hist)[self.few_idx].sum() / hist.sum(axis=1)[self.few_idx].sum()
        return many_acc, medium_acc, few_acc
    
    def get_class_acc(self, idx_array):
        '''Returns accuracy of given classes.'''
        hist = self.confusion_matrix
        acc = []
        for i in range(len(idx_array)):
            acc.append(np.diag(hist)[idx_array[i]].sum() / hist.sum(axis=1)[idx_array[i]])
        return acc
    
    def reset(self):
        '''Reset confusion matrix to 0'''
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
        
    def get_confusion_matrix(self):
        '''Return confusion matrix'''
        return self.confusion_matrix