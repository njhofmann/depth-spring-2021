import pathlib as pl
from typing import Optional, Tuple, List
import math as m

import pandas as pd
import torch as t
import torch.nn as nn
import torch.optim as o
import torch.utils.data as td

import paths as p
from src import arg_parser as apr, init_model as im, eval_seg_model as es, eval_detection_model as ed, datasets as d
import torchinfo as ti


def get_device():
    device = 'cuda' if t.cuda.is_available() else 'cpu'
    print(f'using {device}')
    return t.device(device)


def save_loss_results(iter_path: pl.Path, iter_losses: List[Tuple[int, float]], epoch_path: pl.Path,
                      epoch_losses: List[Tuple[int, float]]) -> None:
    iter_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(iter_losses, columns=['iteration', 'loss']).to_csv(iter_path, index=False)
    pd.DataFrame(epoch_losses, columns=['epoch', 'loss']).to_csv(epoch_path, index=False)


def eval_model(model: nn.Module, data: td.DataLoader, num_of_classes: int, device) -> None:
    with t.no_grad():
        model.eval()
        if seg_or_bbox:
            results = es.eval_seg_model(model, data, num_of_classes, device)
        else:
            results = ed.eval_detect_model(model, data, device)
        print(results)


def train_and_eval(model: nn.Module, train_data: td.DataLoader, test_data: td.DataLoader, device, epochs: Optional[int],
                   optimizer: o.Optimizer, loss_func: nn.Module, num_of_classes: int, save_model: Optional[pl.Path],
                   iter_eval: int, max_iters: Optional[int]) -> None:
    # either epochs or max_iters is None
    if max_iters is None:
        max_iters = epochs * (m.floor(len(train_data.dataset) / batch_size))

    if epochs is None:
        epochs = m.ceil(max_iters / m.floor(len(train_data.dataset) / batch_size))

    iter_losses = []
    epoch_losses = []
    scheduler_count, iters = 0, 0
    for epoch in range(epochs):

        if iters >= max_iters:
            break

        running_loss = 0
        print(f'epoch {epoch}')
        # TODO redo training cycle with bounding boxes
        model.train()
        for i, batch in enumerate(train_data):
            # zero the parameter gradients, forward prop + backward prop + optimizer
            if seg_or_bbox:
                channels, seg_mask = batch
                channels, seg_mask = channels.to(device), seg_mask.to(device)
                optimizer.zero_grad()
                outputs = model(channels)['out']
                loss = loss_func(outputs, seg_mask)
                loss.backward()
                optimizer.step()
            else:
                channels, bboxes, bbox_labels = batch
                channels = channels.to(device)
                for i in range(len(bboxes)):
                    bboxes[i] = bboxes[i].to(device)
                    bbox_labels[i] = bbox_labels[i].to(device)

                targets = [{'boxes': bboxes[i], 'labels': bbox_labels[i]} for i in range(len(bboxes))]
                losses = model(channels, targets)
                loss = losses['rcnn']  # TODO avg loss, rcnn loss?

            scheduler_count += 1
            iters += 1
            running_loss += loss.item() * batch_size  # avg loss for mini batch * batch size

            if iter_eval > 0 and iters % iter_eval == 0:
                iter_losses.append((iters, loss.item()))
                print(f'iteration: {iters}, loss: {loss}')

            if iters >= max_iters:
                break

        epoch_loss = running_loss / len(train_data.dataset)
        print(f'epoch {epoch}, epoch loss {epoch_loss}')
        epoch_losses.append((epoch, epoch_loss))

    eval_model(model, train_data, num_of_classes, device)
    eval_model(model, test_data, num_of_classes, device)

    iter_loss_path = p.RESULTS_DIRC.joinpath('iter_results.csv')
    epoch_loss_path = p.RESULTS_DIRC.joinpath('epoch_results.csv')
    if save_model is not None:
        model_name = save_model.name.split(".")[0]
        iter_loss_path = p.RESULTS_DIRC.joinpath(f'{model_name}_iter_results.csv')
        epoch_loss_path = p.RESULTS_DIRC.joinpath(f'{model_name}_epoch_results.csv')
        t.save(model.state_dict(), save_model)

    save_loss_results(iter_loss_path, iter_losses, epoch_loss_path, epoch_losses)


def bbox_collate_func(batch):
    """Since each image may have a different number of bounding boxes, we need a custom collate function that tells how
    to combine these tensors of different sizes. We use lists
    :param batch: an iterable of N sets from __getitem__()
    :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
    """

    imgs, bboxes, labels = [], [], []
    for b in batch:
        imgs.append(b[0])
        bboxes.append(b[1])
        labels.append(b[2])

    images = t.stack(imgs, dim=0)
    # bboxes = t.Tensor(bboxes)
    # labels = t.Tensor(labels)
    return images, bboxes, labels,  # tensor (N, 3, W, H), 3 lists of N tensors each


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
    train_dataset, test_dataset = d.load_sun_rgbd_dataset(segmentation_or_box=seg_or_bbox,
                                                          include_rgb=rgb,
                                                          include_depth=depth,
                                                          augment=args.no_augment)
    num_of_classes = train_dataset.CLASS_COUNT
    train_data, test_data = init_data_loaders(train_dataset, test_dataset, batch_size, worker_count, seg_or_bbox)

    save_model = create_model_save_path(args.model_name)

    loss_func = None
    if seg_or_bbox:
        loss_func = nn.CrossEntropyLoss()

    device = get_device()
    model = im.init_model(num_of_classes=train_dataset.CLASS_COUNT,
                          device=device,
                          num_of_channels=train_dataset.channel_count,
                          model=args.model,
                          seg_or_box=seg_or_bbox,
                          depth_conv_config=args.depth_conv_option)

    ti.summary(model=model, input_size=(batch_size, train_dataset.channel_count, *d.INPUT_SHAPE))

    optimizer = o.SGD(model.parameters(), lr=.001, momentum=.9, weight_decay=.0004)
    iter_eval = args.iter_eval
    train_and_eval(model=model,
                   epochs=args.epochs,
                   max_iters=args.iterations,
                   optimizer=optimizer,
                   loss_func=loss_func,
                   train_data=train_data,
                   test_data=test_data,
                   device=device,
                   num_of_classes=train_dataset.CLASS_COUNT,
                   save_model=save_model,
                   iter_eval=iter_eval)
