import torch as t
import torch.nn as nn
import torch.utils.data as d
import torch.optim as o
import models.vgg as mv
import src.sun_rgbd_dataloader as ss
import numpy as np


def get_device():
    device = 'cuda' if t.cuda.is_available() else 'cpu'
    print(f'using {device}')
    return t.device(device)


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def eval_output(label_trues, label_preds, n_class, return_iu=False):
    """Evaluates the given list of predicted semantic segmentation masks with the following metrics
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    if return_iu:
        return acc, acc_cls, mean_iu, fwavacc, iu[freq > 0]
    return acc, acc_cls, mean_iu, fwavacc


def train_and_eval(model: nn.Module, train_data: d.DataLoader, test_data: d.DataLoader, device,
                   epochs: int, optimizer: o.Optimizer, loss_func: nn.Module):
    model.to(device)
    for epoch in range(epochs):
        for i, (channels, seg_mask) in enumerate(train_data):
            channels, seg_mask = channels.to(device), seg_mask.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward prop + backward prop + optimizer
            outputs = model(channels)
            loss = loss_func(outputs, seg_mask)
            loss.backward()
            optimizer.step()

    true_masks, pred_masks = [], []
    with t.no_grad():
        for i, (channels, seg_mask) in enumerate(test_data):
            pred_seg_mask = model(channels)
            true_masks.append(seg_mask)
            pred_masks.append(pred_seg_mask)
    eval_output(true_masks, pred_masks, 37)


if __name__ == '__main__':
    batch_size = 1
    worker_count = 4
    shuffle = True
    train_data = d.DataLoader(dataset=ss.SUNRGBDTrainDataset(True, True, False),
                              shuffle=shuffle,
                              num_workers=worker_count,
                              batch_size=1)
    test_data = d.DataLoader(dataset=ss.SUNRGBDTestDataset(True, True, False),
                             num_workers=worker_count)
    loss_func = nn.CrossEntropyLoss()
    model = mv.vgg16()
    train_and_eval(model=model,
                   epochs=1,
                   optimizer=o.SGD(model.parameters(), lr=.001, momentum=.9),
                   loss_func=loss_func,
                   train_data=train_data,
                   test_data=test_data,
                   device=get_device())
