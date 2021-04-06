import pathlib as pl
from typing import Optional, Tuple, List
import math as m

import pandas as pd
import torch as t
import torch.nn as nn
import torch.optim as o
import torch.utils.data as d

import paths as p
from src import arg_parser as apr, init_model as im, eval_seg_model as es, sun_rgbd_dataset as sr, \
    eval_detection_model as ed
import torchinfo as ti


def get_device():
    device = 'cuda' if t.cuda.is_available() else 'cpu'
    print(f'using {device}')
    return t.device(device)


def adjust_scheduler(optimzer: o.Optimizer, iters: int, max_iters: int) -> o.Optimizer:
    old_lr = optimzer.param_groups[0]['lr']
    new_lr = max(1e-6, old_lr * ((1 - (iters / max_iters)) ** .9))
    optimzer.param_groups[0]['lr'] = new_lr
    return optimzer


def save_loss_results(iter_path: pl.Path, iter_losses: List[Tuple[int, float]], epoch_path: pl.Path,
                      epoch_losses: List[Tuple[int, float]]) -> None:
    iter_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(iter_losses, columns=['iteration', 'loss']).to_csv(iter_path, index=False)
    pd.DataFrame(epoch_losses, columns=['epoch', 'loss']).to_csv(epoch_path, index=False)


def train_and_eval(model: nn.Module, train_data: d.DataLoader, test_data: d.DataLoader, device, epochs: Optional[int],
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
        for i, (channels, seg_mask) in enumerate(train_data):
            channels, seg_mask = channels.to(device), seg_mask.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward prop + backward prop + optimizer
            outputs = model(channels)['out']
            loss = loss_func(outputs, seg_mask)
            loss.backward()
            optimizer.step()

            scheduler_count += 1
            iters += 1
            running_loss += loss.item() * outputs.shape[0]  # avg loss for mini batch * batch size

            if iter_eval > 0 and iters % iter_eval == 0:
                iter_losses.append((iters, loss.item()))
                print(f'iteration: {iters}, loss: {loss}')

            if iters >= max_iters:
                break

            if scheduler_count == 10:
                scheduler_count = 0
                #optimizer = adjust_scheduler(optimizer, iters, max_iters)

        epoch_loss = running_loss / len(train_data.dataset)
        print(f'epoch {epoch}, epoch loss {epoch_loss}')
        epoch_losses.append((epoch, epoch_loss))

    with t.no_grad():
        model.eval()
        # TODO object detection eval here
        if segmentation_or_box:
            results = es.eval_seg_model(model, test_data, num_of_classes, device)
        else:
            results = ed.eval_det_model(model, test_data)
        print()

    iter_loss_path = p.RESULTS_DIRC.joinpath('iter_results.csv')
    epoch_loss_path = p.RESULTS_DIRC.joinpath('epoch_results.csv')
    if save_model is not None:
        model_name = save_model.name.split(".")[0]
        iter_loss_path = p.RESULTS_DIRC.joinpath(f'{model_name}_iter_results.csv')
        epoch_loss_path = p.RESULTS_DIRC.joinpath(f'{model_name}_epoch_results.csv')
        t.save(model.state_dict(), save_model)

    save_loss_results(iter_loss_path, iter_losses, epoch_loss_path, epoch_losses)


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


def create_model_save_path(model_name: Optional[str]) -> Optional[pl.Path]:
    path = None if model_name is None else p.TRAINED_MODELS_DIRC.joinpath(f'{model_name}.pt')
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


if __name__ == '__main__':
    args = apr.get_user_args()
    rgb, depth = args.channels
    segmentation_or_box = args.seg
    batch_size = args.batch_size
    train_dataset, test_dataset = sr.load_sun_rgbd_dataset(segmentation_or_box=segmentation_or_box,
                                                           include_rgb=rgb,
                                                           include_depth=depth,
                                                           augment=args.no_augment)
    num_of_classes = train_dataset.CLASS_COUNT
    train_data, test_data = init_data_loaders(train_dataset, test_dataset, batch_size)

    save_model = create_model_save_path(args.model_name)

    loss_func = None
    if segmentation_or_box:
        loss_func = nn.CrossEntropyLoss()

    device = get_device()
    model = im.init_model(num_of_classes=train_dataset.CLASS_COUNT,
                          device=device,
                          num_of_channels=train_dataset.channel_count,
                          model=args.model,
                          seg_or_box=segmentation_or_box)

    ti.summary(model=model, input_size=(batch_size, train_dataset.channel_count, *sr.INPUT_SHAPE))

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
