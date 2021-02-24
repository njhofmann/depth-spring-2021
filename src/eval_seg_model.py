import torch as t
import torch.nn as nn
import torch.utils.data as d
import torch.optim as o
import models.vgg as mv
import src.sun_rgbd_dataloader as ss


def get_device():
    return t.device('cuda' if t.cuda.is_available() else 'cpu')


def train_and_eval(model: nn.Module, train_data: d.DataLoader, test_data: d.DataLoader, device,
                   epochs: int, optimizer: o.Optimizer, loss: nn.Module):
    model.to(device)
    for epoch in range(epochs):
        for i, (channels, seg_mask) in enumerate(train_data):
            channels, seg_mask = channels.to(device), seg_mask.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(channels)
            loss = loss(outputs, seg_mask)
            loss.backward()
            optimizer.step()


if __name__ == '__main__':
    batch_size = 1
    worker_count = 4
    shuffle = True
    train_data = d.DataLoader(dataset=ss.SUNRGBDTrainDataset(True, True, True),
                              shuffle=shuffle,
                              num_workers=worker_count,
                              batch_size=1)
    test_data = d.DataLoader(dataset=ss.SUNRGBDTestDataset(True, True, True),
                             num_workers=worker_count)
    loss_func = nn.CrossEntropyLoss()
    model = mv.vgg16()
    train_and_eval(model=model,
                   epochs=10,
                   optimizer=o.SGD(model.parameters(), lr=.001, momentum=.9),
                   loss=loss_func,
                   train_data=train_data,
                   test_data=test_data,
                   device=get_device())
