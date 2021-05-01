from typing import Tuple, Optional

from torch import nn as nn
import torch as t
from torchvision.models import _utils as su
import torchvision.models.detection.anchor_utils as au
import torchvision.ops as to
from torchvision.models.segmentation import deeplabv3 as dl
import torchvision.models.detection as td

import models as m
import src.multi_gpu_faster_rcnn as mg


def init_backbone(model: str, channel_cnt: int, depth_conv_alpha: float,
                  depth_conv_option: Optional[str] = None) -> Tuple[nn.Module, int]:
    # rgbd data will be split into rgb and depth data
    if depth_conv_option is not None:
        channel_cnt = 3

    if model == 'vgg':
        in_channel_cnt = 512
        base_model = m.vgg(in_channels=channel_cnt,
                           depth_conv_option=depth_conv_option,
                           depth_conv_alpha=depth_conv_alpha)
        return base_model, in_channel_cnt
    elif model == 'resnet':
        in_channel_cnt = 2048
        model = m.resnet(input_channels=channel_cnt,
                         depth_conv_option=depth_conv_option,
                         depth_conv_alpha=depth_conv_alpha)
        return model, in_channel_cnt
    # elif model == 'densenet':
    #     depth_conv_config = get_densenet_depth_conv_config(depth_conv_option)
    #     in_channel_cnt = 1664
    #     base_model = m.densenet(input_channels=channel_cnt, depth_conv_config=depth_conv_config)
    #     return_layers = {'features': 'out'}
    elif model == 'alexnet':
        backbone = m.alexnet(in_channels=channel_cnt,
                             depth_conv_option=depth_conv_option,
                             depth_conv_alpha=depth_conv_alpha)
        return backbone, 256
    else:
        raise ValueError(f'model {model} is an unsupported model')


def init_model(num_of_classes: int, num_of_channels: int, model: str, seg_or_box: bool, device,
               depth_conv_alpha: int, depth_conv_config: Optional[str] = None) -> nn.Module:
    backbone, in_channels = init_backbone(model, num_of_channels, depth_conv_alpha, depth_conv_config)
    if seg_or_box:
        # generated from https://pytorch.org/vision/0.8/_modules/torchvision/models/segmentation/segmentation.html
        classifier = dl.DeepLabHead(in_channels=in_channels, num_classes=num_of_classes)
        model = dl.DeepLabV3(backbone=backbone, classifier=classifier)
        return model.to(device)
    else:
        # generated from
        # https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
        anchor_generator = au.AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                              aspect_ratios=((0.5, 1.0, 2.0),))
        roi_pooler = to.MultiScaleRoIAlign(featmap_names=['out'],
                                           output_size=7,
                                           sampling_ratio=2)
        backbone.out_channels = in_channels

        # hack around internal normalization
        return mg.MultiGPUFasterRCNN(backbone=backbone,
                                     rpn_anchor_generator=anchor_generator,
                                     num_classes=num_of_classes,
                                     box_roi_pool=roi_pooler,
                                     image_mean=t.Tensor([0. for _ in range(num_of_channels)]),
                                     image_std=t.Tensor([1. for _ in range(num_of_channels)]))


if __name__ == '__main__':
    init_backbone('densenet', 3)
