from typing import Tuple, Optional

from torch import nn as nn
import torch as t
from torchvision.models import _utils as su
from torchvision.models.segmentation import deeplabv3 as dl
import torchvision.models.detection as td

import models as m
import src.multi_gpu_faster_rcnn as mg


def base_depth_conv_config(option: str, start_config: dict, end_config: dict, all_config: dict) -> dict:
    if option == 'all':
        return all_config
    elif option == 'start':
        return start_config
    elif option == 'end':
        return end_config
    elif option is None:
        return None
    raise ValueError(f'{option} is an invalid depth convolutional option')


def get_vgg_depth_conv_config(option: str) -> dict:
    return base_depth_conv_config(option,
                                  all_config=None,
                                  start_config=None,
                                  end_config=None)


def get_resnet_depth_conv_config(option: str) -> dict:
    return base_depth_conv_config(option,
                                  all_config=None,
                                  start_config=None,
                                  end_config=None)


def get_densenet_depth_conv_config(option: str) -> dict:
    return base_depth_conv_config(option,
                                  all_config=None,
                                  start_config=None,
                                  end_config=None)


def init_backbone(model: str, channel_cnt: int, depth_conv_option: Optional[str] = None) -> Tuple[nn.Module, int]:
    if model == 'vgg':
        depth_conv_config = get_vgg_depth_conv_config(depth_conv_option)
        depth_conv_config = depth_conv_config  # TODO add me later
        base_model = m.vgg(pretrained=False, in_channels=channel_cnt)
        in_channel_cnt = 512
        return_layers = {'features': 'out'}
    elif model == 'resnet':
        depth_conv_config = get_resnet_depth_conv_config(depth_conv_option)
        in_channel_cnt = 512  # 2048
        depth_conv_config = depth_conv_config  # TODO add me later
        base_model = m.resnet2()  # (input_channels=channel_cnt)
        return_layers = {'layer4': 'out'}
    elif model == 'densenet':
        depth_conv_config = get_resnet_depth_conv_config(depth_conv_option)
        in_channel_cnt = 1664
        base_model = m.densenet(input_channels=channel_cnt, depth_conv_config=depth_conv_config)
        return_layers = {'features': 'out'}
    else:
        raise ValueError(f'model {model} is an unsupported model')

    backbone = su.IntermediateLayerGetter(base_model, return_layers)
    return backbone, in_channel_cnt


def init_model(num_of_classes: int, num_of_channels: int, model: str, seg_or_box: bool, device,
               depth_conv_config: Optional[str] = None) -> nn.Module:
    backbone, in_channels = init_backbone(model, num_of_channels, depth_conv_config)
    if seg_or_box:
        # generated from
        # https://pytorch.org/vision/0.8/_modules/torchvision/models/segmentation/segmentation.html
        classifier = dl.DeepLabHead(in_channels=in_channels, num_classes=num_of_classes)
        model = dl.DeepLabV3(backbone=backbone, classifier=classifier)
    else:
        # generated from
        # https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
        # anchor_generator = rpn.AnchorGenerator(sizes=((64, 128, 256),),
        #                                        aspect_ratios=((0.5, 1.0,),))
        # roi_pooler = to.MultiScaleRoIAlign(featmap_names=['out'],
        #                                    output_size=3,
        #                                    sampling_ratio=2)
        backbone.out_channels = in_channels

        # hack around internal normalization
        model = mg.MultiGPUFasterRCNN(backbone=backbone,
                                      num_classes=num_of_classes,
                                      image_mean=t.Tensor([0. for _ in range(num_of_channels)]),
                                      image_std=t.Tensor([1. for _ in range(num_of_channels)]))
    return model.to(device)


if __name__ == '__main__':
    init_backbone('densenet', 3)
