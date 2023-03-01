import numpy as np
# import matplotlib.pyplot as plt

def _fast_hist(label_true, label_pred, n_class):
    '''
    Returns: Confusion Matrix
    '''
    # mask - boolean ndarray
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

def calc_IoU(label_preds, label_trues, n_class):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    # acc = np.diag(hist).sum() / hist.sum()
    # with np.errstate(divide='ignore', invalid='ignore'):
    #     acc_cls = np.diag(hist) / hist.sum(axis=1)
    # acc_cls = np.nanmean(acc_cls)
    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (
            hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        )
    mean_iu = np.nanmean(iu)
    # freq = hist.sum(axis=1) / hist.sum()
    # fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    return mean_iu


class MetricLogger(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.many_idx = [1, 5, 8, 9]
        # self.medium_idx = [6, 7]
        # self.few_idx = [2, 3, 4]
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
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))
        
        return mean_cls, mean_iu, acc_cls, acc

    def get_acc_cat(self):
            """Returns accuracy of different categories.
            """
            hist = self.confusion_matrix
            many_acc = np.diag(hist)[self.many_idx].sum() / hist.sum(axis=1)[self.many_idx].sum()
            medium_acc = np.diag(hist)[self.medium_idx].sum() / hist.sum(axis=1)[self.medium_idx].sum()
            few_acc = np.diag(hist)[self.few_idx].sum() / hist.sum(axis=1)[self.few_idx].sum()
            return many_acc, medium_acc, few_acc
    
    def get_class_acc(self, idx_array):
        hist = self.confusion_matrix
        acc = []
        for i in range(len(idx_array)):
            acc.append(np.diag(hist)[idx_array[i]].sum() / hist.sum(axis=1)[idx_array[i]])

        return acc
    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
        
    def get_confusion_matrix(self):
        return self.confusion_matrix