import pathlib as pl
from typing import Optional, Tuple, List

import pandas as pd
import torch as t
import torch.nn as nn
import torch.optim as o
import torch.utils.data as d

import paths as p
from src import arg_parser as apr, init_model as im, eval as e, sun_rgbd_dataset as sr
import torchinfo as ti


def get_device():
    device = 'cuda' if t.cuda.is_available() else 'cpu'
    print(f'using {device}')
    return t.device(device)


def cuda_tensor_to_np_arr(tensor):
    return tensor.cpu().numpy()[0]


def adjust_scheduler(optimzer: o.Optimizer, iters: int, max_iters: int) -> o.Optimizer:
    optimzer.param_groups[0]['lr'] *= (1 - (iters / max_iters)) ** .9
    return optimzer


def save_loss_results(path: pl.Path, losses: List[Tuple[int, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(losses, columns=['iterations', 'loss']).to_csv(path, index=False)


def train_and_eval(model: nn.Module, train_data: d.DataLoader, test_data: d.DataLoader, device, epochs: int,
                   optimizer: o.Optimizer, loss_func: nn.Module, num_of_classes: int, save_model: Optional[pl.Path],
                   iter_eval: int) -> None:
    losses = []
    scheduler_count, iters = 0, 0
    max_iters = epochs * len(train_data)
    for epoch in range(epochs):
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

            if iter_eval > 0 and iters % iter_eval == 0:
                losses.append((iters, loss.data[0]))
                print(f'iteration: {iters}, loss: {loss}')

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
            results_hist = e.build_hist(cuda_tensor_to_np_arr(seg_mask),
                                        cuda_tensor_to_np_arr(pred_seg_mask),
                                        results_hist,
                                        num_of_classes)
    print(e.eval_results(results_hist))
    loss_results_path = p.RESULTS_DIRC.joinpath('results.csv')
    if save_model is not None:
        loss_results_path = loss_results_path.parent.joinpath(f'{save_model.name.split(".")[0]}_results.csv')
        t.save(model.state_dict(), save_model)

    save_loss_results(loss_results_path, losses)


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
        path.mkdir(parents=True, exist_ok=True)
    return path


if __name__ == '__main__':
    args = apr.get_user_args()
    rgb, depth = args.channels
    semantic_or_box = args.seg
    batch_size = args.batch_size
    train_dataset, test_dataset = sr.load_sun_rgbd_dataset(semantic_or_box=semantic_or_box,
                                                           include_rgb=rgb,
                                                           include_depth=depth)
    num_of_classes = train_dataset.CLASS_COUNT
    train_data, test_data = init_data_loaders(train_dataset, test_dataset, batch_size)

    save_model = create_model_save_path(args.model_name)

    loss_func = nn.CrossEntropyLoss()
    device = get_device()
    model = im.init_model(num_of_classes=train_dataset.CLASS_COUNT,
                          device=device,
                          num_of_channels=train_dataset.channel_count,
                          model=args.model)

    ti.summary(model=model, input_size=(batch_size, train_dataset.channel_count, *sr.INPUT_SHAPE))

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
                   save_model=save_model,
                   iter_eval=iter_eval)
