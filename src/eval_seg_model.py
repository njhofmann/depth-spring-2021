from typing import Optional

import numpy as np
import torch.nn as nn
import torch as t

import src.util as u


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
    return {'accuracy': acc, 'class accuracy': acc_cls, 'mean iu': mean_iu, 'fwav accuracy': fwavacc}


def eval_seg_model(model: nn.Module, test_data, num_of_classes: int, device):
    results_hist = None
    for i, batch in enumerate(test_data):
        depth_channel = None
        if len(batch) == 2:
            channels, seg_mask = batch
        else:
            channels, depth_channel, seg_mask = batch
            depth_channel = depth_channel.to(device)
        channels, seg_mask = channels.to(device), seg_mask.to(device)

        pred_seg_mask = t.argmax(t.softmax(model(channels, depth_channel)['out'], dim=1), dim=1)
        results_hist = build_hist(u.cuda_tensor_to_np_arr(seg_mask),
                                  u.cuda_tensor_to_np_arr(pred_seg_mask),
                                  results_hist,
                                  num_of_classes)
    return eval_results(results_hist)
