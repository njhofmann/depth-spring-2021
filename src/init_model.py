from typing import Tuple

from torch import nn as nn
from torchvision.models import _utils as su
from torchvision.models.segmentation import deeplabv3 as dl

import models as m


def init_backbone(model: str, channel_cnt: int) -> Tuple[nn.Module, int]:
    if model == 'vgg':
        base_model = m.vgg(pretrained=False, in_channels=channel_cnt)
        in_channel_cnt = 512
        return_layers = {'features': 'out'}
    elif model == 'resnet':
        in_channel_cnt = 2048
        base_model = m.resnet(input_channels=channel_cnt)
        return_layers = {'layer4': 'out'}
    elif model == 'densenet':
        in_channel_cnt = 1664
        base_model = m.densenet(input_channels=channel_cnt)
        return_layers = {'features': 'out'}
    else:
        raise ValueError(f'model {model} is an unsupported model')

    backbone = su.IntermediateLayerGetter(base_model, return_layers)
    return backbone, in_channel_cnt


def init_model(num_of_classes: int, num_of_channels: int, model: str, device):
    # copied init process from
    # https://pytorch.org/vision/0.8/_modules/torchvision/models/segmentation/segmentation.html
    backbone, in_channels = init_backbone(model, num_of_channels)
    classifier = dl.DeepLabHead(in_channels=in_channels, num_classes=num_of_classes)
    seg_model = dl.DeepLabV3(backbone=backbone, classifier=classifier)
    return seg_model.to(device)


if __name__ == '__main__':
    init_backbone('densenet', 3)