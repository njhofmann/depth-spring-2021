import pathlib as pl
from typing import Optional, Tuple, List, Dict, Union
import math as m
import collections as c

import pandas as pd
import torch as t
import torch.cuda
import torch.nn as nn
import torch.optim as o
import torch.utils.data as td

import paths as p
from src import arg_parser as apr, init_model as im, eval_seg_model as es, eval_detection_model as ed, datasets as d
import torchinfo as ti


def get_and_set_default_device(default_gpu: int):
    if t.cuda.is_available():
        torch.cuda.set_device(default_gpu)
        device = 'cuda'
    else:
        device = 'cpu'
    print(f'using {device}')
    return t.device(device)


def separate_loss_info(loss_info: List[Tuple[int, Dict[str, float]]]) \
        -> Tuple[List[str], List[Tuple[Union[int, float]]]]:
    cols = list(loss_info[0][1].keys())
    losses = [(step, *[losses[col] for col in cols]) for step, losses in loss_info]
    return cols, losses


def save_loss_results(iter_path: pl.Path, iter_losses: List[Tuple[int, Dict[str, float]]], epoch_path: pl.Path,
                      epoch_losses: List[Tuple[int, Dict[str, float]]]) -> None:
    iter_path.parent.mkdir(parents=True, exist_ok=True)

    iter_cols, iter_data = separate_loss_info(iter_losses)
    pd.DataFrame(iter_data, columns=['iteration', *iter_cols]).to_csv(iter_path, index=False)

    epoch_cols, epoch_data = separate_loss_info(epoch_losses)
    pd.DataFrame(epoch_data, columns=['epoch', *epoch_cols]).to_csv(epoch_path, index=False)


def eval_model(model: nn.Module, data: td.DataLoader, num_of_classes: int, device) -> None:
    with t.no_grad():
        model.eval()
        if seg_or_bbox:
            results = es.eval_seg_model(model, data, num_of_classes, device)
        else:
            results = ed.eval_detect_model(model, data, device)
        print(results)


def train_and_eval(model: nn.Module, train_data: td.DataLoader, test_data: td.DataLoader, device, epochs: Optional[int],
                   num_of_classes: int, save_model: Optional[pl.Path], iter_eval: int, max_iters: Optional[int]) -> None:
    # either epochs or max_iters is None
    if max_iters is None:
        max_iters = epochs * (m.floor(len(train_data.dataset) / batch_size))

    if epochs is None:
        epochs = m.ceil(max_iters / m.floor(len(train_data.dataset) / batch_size))

    if seg_or_bbox:
        optimizer = o.SGD(model.parameters(), lr=.001, momentum=.9, weight_decay=.0004)
        loss_func = nn.CrossEntropyLoss()
    else:
        optimizer, loss_func = None, None

    iter_losses = []
    cum_losses = []
    iters = 0
    for epoch in range(epochs):

        if iters >= max_iters:
            break

        running_loss = c.Counter()
        print(f'epoch {epoch}')
        # TODO redo training cycle with bounding boxes
        model.train()
        for i, batch in enumerate(train_data):
            # zero the parameter gradients, forward prop + backward prop + optimizer
            if seg_or_bbox:
                depth_channels = None
                if len(batch) == 2:
                    channels, seg_mask = batch
                else:
                    print(5)
                    channels, depth_channels, seg_mask = batch
                    depth_channels = depth_channels.to(device)
                channels, seg_mask = channels.to(device), seg_mask.to(device)

                optimizer.zero_grad()
                outputs = model(channels, depth_channels)['out']

                loss = loss_func(outputs, seg_mask)
                loss.backward()
                optimizer.step()
                losses = {'loss': loss.item()}
                running_loss['loss'] += loss.item() * batch_size
            else:
                channels, bboxes, bbox_labels = batch
                channels = channels.to(device)
                for i in range(len(bboxes)):
                    bboxes[i] = bboxes[i].to(device)
                    bbox_labels[i] = bbox_labels[i].to(device)

                targets = [{'boxes': bboxes[i], 'labels': bbox_labels[i]} for i in range(len(bboxes))]
                losses = model(channels, targets)
                losses = {'classifier loss': losses['loss_classifier'].item(),
                          'box regression loss': losses['loss_box_reg'].item(),
                          'objectness loss': losses['loss_objectness'].item(),
                          'rpn box regression loss': losses['loss_rpn_box_reg']}

                for k, v in losses.items():
                    running_loss[k] += v

            iters += 1
            # running_loss += loss.item() * batch_size  # avg loss for mini batch * batch size

            if iter_eval > 0 and iters % iter_eval == 0:
                iter_losses.append((iters, losses))
                print(f'iteration: {iters}, ' + ', '.join([f'{k}: {v}' for k, v in losses.items()]))

            if iters >= max_iters:
                break

        # epoch_loss = running_loss / len(train_data.dataset)
        epoch_losses = {k: v / len(train_data.dataset) for k, v in running_loss.items()}
        print(f'epoch {epoch}, ' + ', '.join([f'{k}: {v}' for k, v in epoch_losses.items()]))
        cum_losses.append((epoch, epoch_losses))

    eval_model(model, train_data, num_of_classes, device)
    eval_model(model, test_data, num_of_classes, device)

    iter_loss_path = p.RESULTS_DIRC.joinpath('iter_results.csv')
    epoch_loss_path = p.RESULTS_DIRC.joinpath('epoch_results.csv')
    if save_model is not None:
        model_name = save_model.name.split(".")[0]
        iter_loss_path = p.RESULTS_DIRC.joinpath(f'{model_name}_iter_results.csv')
        epoch_loss_path = p.RESULTS_DIRC.joinpath(f'{model_name}_epoch_results.csv')
        t.save(model.state_dict(), save_model)

    save_loss_results(iter_loss_path, iter_losses, epoch_loss_path, cum_losses)


def bbox_collate_func(batch):
    """Since each image may have a different number of bounding boxes, we need a custom collate function that tells how
    to combine these tensors of different sizes.
    :param batch: an iterable of N sets from __getitem__()
    :return: a tensor of images, lists of varying-size tensors of bounding boxes and their class labels
    """
    imgs, bboxes, bbox_labels = list(zip(*batch))
    images = t.stack(imgs, dim=0)
    return images, list(bboxes), list(bbox_labels),  # tensor (N, 3, W, H), 3 lists of N tensors each


def init_data_loaders(train_set, test_set, batch_size: int, worker_count: int, seg_or_bbox: bool) \
        -> Tuple[td.DataLoader, td.DataLoader]:
    collate_func = None if seg_or_bbox else bbox_collate_func
    shuffle = True
    train = td.DataLoader(dataset=train_set,
                          shuffle=shuffle,
                          num_workers=worker_count,
                          batch_size=batch_size,
                          drop_last=True,
                          collate_fn=collate_func)
    test = td.DataLoader(dataset=test_set,
                         num_workers=worker_count,
                         batch_size=batch_size,
                         collate_fn=collate_func)
    return train, test


def create_model_save_path(model_name: Optional[str]) -> Optional[pl.Path]:
    path = None if model_name is None else p.TRAINED_MODELS_DIRC.joinpath(f'{model_name}.pt')
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


if __name__ == '__main__':
    args = apr.get_user_args()
    rgb, depth = args.channels
    seg_or_bbox = args.seg
    batch_size = args.batch_size
    worker_count = 4 * t.cuda.device_count()
    depth_conv_option = args.depth_conv_option
    sep_rgbd = depth_conv_option is not None
    train_dataset, test_dataset = d.load_sun_rgbd_dataset(segmentation_or_box=seg_or_bbox,
                                                          include_rgb=rgb,
                                                          include_depth=depth,
                                                          augment=args.no_augment,
                                                          sep_rgbd=sep_rgbd)
    num_of_classes = train_dataset.CLASS_COUNT
    train_data, test_data = init_data_loaders(train_dataset, test_dataset, batch_size, worker_count, seg_or_bbox)

    save_model = create_model_save_path(args.model_name)

    device = get_and_set_default_device(args.set_device)
    model = im.init_model(num_of_classes=train_dataset.CLASS_COUNT,
                          device=device,
                          num_of_channels=train_dataset.channel_count,
                          model=args.model,
                          seg_or_box=seg_or_bbox,
                          depth_conv_config=depth_conv_option,
                          depth_conv_alpha=args.alpha)

    # does not work for multi-gpu models, or depth convs
    if seg_or_bbox and args.depth_conv_option is None:
        ti.summary(model=model.backbone, input_size=(batch_size, train_dataset.channel_count, *d.INPUT_SHAPE))
    else:
        print(model.backbone)

    iter_eval = args.iter_eval
    train_and_eval(model=model,
                   epochs=args.epochs,
                   max_iters=args.iterations,
                   train_data=train_data,
                   test_data=test_data,
                   device=device,
                   num_of_classes=train_dataset.CLASS_COUNT,
                   save_model=save_model,
                   iter_eval=iter_eval)
