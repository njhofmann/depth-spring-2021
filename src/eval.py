from typing import Optional

import numpy as np


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    return np.bincount(n_class * label_true[mask].astype(int) + label_pred[mask],
                       minlength=n_class ** 2).reshape(n_class, n_class)


def build_hist(true_label, pred_label, hist: Optional[np.ndarray] = None, n_class: Optional[int] = None):
    if hist is None:
        hist = np.zeros((n_class, n_class))
    return hist + _fast_hist(true_label.flatten(), pred_label.flatten(), n_class)


def eval_results(hist, return_iu: bool = False):
    """Evaluates the given list of predicted semantic segmentation masks with the following metrics
      - overall accuracy
      - mean accuracy per class
      - mean IU
      - fwavacc
    """
    # label_trues, label_preds, n_class, return_iu=False
    # hist = np.zeros((n_class, n_class))
    # for lt, lp in zip(label_trues, label_preds):
    #     hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    if return_iu:
        return acc, acc_cls, mean_iu, fwavacc, iu[freq > 0]
    # TODO fix names
    return {'accuracy': acc, 'class accuracy': acc_cls, 'mean iu': mean_iu, 'fwav accuracy': fwavacc}