import torch as t
import torch.nn as nn
import torch.utils.data as d
import torch.optim as o
from models import vgg as mv, resnet as mr, densenet as md
import src.sun_rgbd_dataset as ss
import numpy as np
from typing import Tuple, Optional
import torchvision.models.vgg as vgg
import torchvision.models._utils as su
import torchvision.models.segmentation.deeplabv3 as dl


def init_backbone(model: str, channel_cnt: int) -> nn.Module:
    if model == 'vgg':
        backbone = mv.vgg16(pretrained=False, in_channels=channel_cnt)  # mv.vgg16(pretrained=False)
        in_features = backbone.classifier[6].in_features
        backbone.classifier[6] = nn.Linear(in_features, in_features)
        return_layers = {'features': 'out'}
        return su.IntermediateLayerGetter(backbone, return_layers=return_layers)
    elif model == 'resnet':
        backbone = mr.resnet50(input_channels=channel_cnt)
        return su.IntermediateLayerGetter(backbone, {'layer4': 'out'})
    elif model == 'densenet':
        pass
    raise ValueError(f'model {model} is an unsupported model')


def get_device():
    device = 'cuda' if t.cuda.is_available() else 'cpu'
    print(f'using {device}')
    return t.device(device)


def init_model(num_of_classes: int, num_of_channels: int, model: str, device):
    # copied init process from
    # https://pytorch.org/vision/0.8/_modules/torchvision/models/segmentation/segmentation.html
    backbone = init_backbone(model, num_of_channels)
    classifier = dl.DeepLabHead(in_channels=512, num_classes=num_of_classes)
    seg_model = dl.DeepLabV3(backbone=backbone, classifier=classifier)
    print(seg_model)
    seg_model = seg_model.to(device)
    return seg_model


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
      - mean accuracy
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
    return acc, acc_cls, mean_iu, fwavacc


def cuda_tensor_to_np_arr(tensor):
    return tensor.cpu().numpy()[0]


def train_and_eval(model: nn.Module, train_data: d.DataLoader, test_data: d.DataLoader, device,
                   epochs: int, optimizer: o.Optimizer, loss_func: nn.Module, num_of_classes: int):
    for epoch in range(epochs):
        print(f'epoch {epoch}')
        for i, (channels, seg_mask) in enumerate(train_data):
            channels, seg_mask = channels.to(device), seg_mask.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward prop + backward prop + optimizer
            outputs = t.softmax(model(channels)['out'], dim=1)
            loss = loss_func(outputs, seg_mask)
            loss.backward()
            optimizer.step()

    with t.no_grad():
        model.eval()
        results_hist = None
        for i, (channels, seg_mask) in enumerate(test_data):
            channels, seg_mask = channels.to(device), seg_mask.to(device)
            pred_seg_mask = t.argmax(t.softmax(model(channels)['out'], dim=1), dim=1)
            results_hist = build_hist(cuda_tensor_to_np_arr(seg_mask),
                                      cuda_tensor_to_np_arr(pred_seg_mask),
                                      results_hist,
                                      num_of_classes)
    print(eval_results(results_hist))


if __name__ == '__main__':
    rgb, depth = True, False
    batch_size = 16
    epochs = 0
    worker_count = 4
    shuffle = True
    train_dataset, test_dataset = ss.load_sun_rgbd_dataset(True, include_rgb=False, include_depth=True)
    num_of_classes = train_dataset.CLASS_COUNT
    train_data = d.DataLoader(dataset=train_dataset,
                              shuffle=shuffle,
                              num_workers=worker_count,
                              batch_size=batch_size,
                              drop_last=True)
    test_data = d.DataLoader(dataset=test_dataset,
                             num_workers=worker_count)
    loss_func = nn.CrossEntropyLoss()
    device = get_device()
    model = init_model(num_of_classes=train_dataset.CLASS_COUNT,
                       device=device,
                       num_of_channels=train_dataset.channel_count,
                       model='resnet')
    optimizer = o.SGD(model.parameters(), lr=.001, momentum=.9)
    train_and_eval(model=model,
                   epochs=epochs,
                   optimizer=optimizer,
                   loss_func=loss_func,
                   train_data=train_data,
                   test_data=test_data,
                   device=device,
                   num_of_classes=num_of_classes)
