import pathlib as pl
from typing import Optional, Tuple

import torch as t
import torch.nn as nn
import torch.optim as o
import torch.utils.data as d
import torchvision.models._utils as su
import torchvision.models.segmentation.deeplabv3 as dl

import models as m
import src.arg_parser as apr
import src.sun_rgbd_dataset as ss
from src.eval import build_hist, eval_results


def init_backbone(model: str, channel_cnt: int) -> nn.Module:
    if model == 'vgg':
        backbone = m.vgg(pretrained=False, in_channels=channel_cnt)  # mv.vgg16(pretrained=False)
        # in_features = backbone.classifier[6].in_features
        # backbone.classifier[6] = nn.Linear(in_features, in_features)
        return su.IntermediateLayerGetter(backbone, return_layers={'features': 'out'})
    elif model == 'resnet':
        backbone = m.resnet(input_channels=channel_cnt)
        return su.IntermediateLayerGetter(backbone, return_layers={'layer4': 'out'})
    elif model == 'densenet':
        backbone = m.densenet(input_channels=channel_cnt)
        return su.IntermediateLayerGetter(backbone, return_layers={'features': 'out'})
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


def cuda_tensor_to_np_arr(tensor):
    return tensor.cpu().numpy()[0]


def adjust_scheduler(optimzer: o.Optimizer, iters: int, max_iters: int) -> o.Optimizer:
    optimzer.param_groups[0]['lr'] *= (1 - (iters / max_iters)) ** .9
    return optimzer


def train_and_eval(model: nn.Module, train_data: d.DataLoader, test_data: d.DataLoader, device, epochs: int,
                   optimizer: o.Optimizer, loss_func: nn.Module, num_of_classes: int, save_model: Optional[pl.Path],
                   iter_eval: int) \
        -> None:
    losses = []
    scheduler_count, iters = 0, 0
    max_iters = epochs * len(train_data)
    for epoch in range(epochs):
        print(f'epoch {epoch}')
        for i, (channels, seg_mask) in enumerate(train_data):
            print(f'iteration {i}')
            channels, seg_mask = channels.to(device), seg_mask.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward prop + backward prop + optimizer
            outputs = model(channels)['out']
            loss = loss_func(outputs, seg_mask)
            loss.backward()
            optimizer.step()

            if iter_eval > 0 and iters % iter_eval == 0:
                losses.append((iters, loss))
                print(f'loss: {loss}')

            scheduler_count += 1
            iters += 1
            if scheduler_count == 10:
                scheduler_count = 0
                optimizer = adjust_scheduler(optimizer, iters, max_iters)

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

    if save_model is not None:
        t.save(model.state_dict(), save_model)


def init_data_loaders(train_set, test_set, batch_size: int) -> Tuple[d.DataLoader, d.DataLoader]:
    shuffle = True
    workers = 4
    train_data = d.DataLoader(dataset=train_set,
                              shuffle=shuffle,
                              num_workers=workers,
                              batch_size=batch_size,
                              drop_last=True)
    test_data = d.DataLoader(dataset=test_set,
                             num_workers=workers,
                             batch_size=batch_size)
    return train_data, test_data


if __name__ == '__main__':
    args = apr.get_user_args()
    rgb, depth = args.channels
    semantic_or_box = args.seg
    train_dataset, test_dataset = ss.load_sun_rgbd_dataset(semantic_or_box=semantic_or_box,
                                                           include_rgb=rgb,
                                                           include_depth=depth)
    num_of_classes = train_dataset.CLASS_COUNT
    train_data, test_data = init_data_loaders(train_dataset, test_dataset, args.batch_size)

    loss_func = nn.CrossEntropyLoss()
    device = get_device()
    model = init_model(num_of_classes=train_dataset.CLASS_COUNT,
                       device=device,
                       num_of_channels=train_dataset.channel_count,
                       model=args.model)
    optimizer = o.SGD(model.parameters(), lr=.001, momentum=.9)
    iter_eval = args.iter_eval
    train_and_eval(model=model,
                   epochs=args.epochs,
                   optimizer=optimizer,
                   loss_func=loss_func,
                   train_data=train_data,
                   test_data=test_data,
                   device=device,
                   num_of_classes=train_dataset.CLASS_COUNT,
                   save_model=None,
                   iter_eval=iter_eval)
