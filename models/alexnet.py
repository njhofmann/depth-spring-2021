import torch
import torch.nn as nn
from .utils import load_state_dict_from_url
from typing import Any, Optional, List, Tuple, Union
import depth_conv_ops.depthaware.models.ops.depthconv.module as dc
import models.utils as mu

__all__ = ['AlexNet', 'alexnet']

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


def conv(in_planes: int, out_planes: int, kernel_size: int = 3, stride: int = 1, groups: int = 1, dilation: int = 1,
         padding: int = 0, depth_conv_alpha: float = 8.3, depth_conv: bool = False) -> Union[nn.Conv2d, dc.DepthConv]:
    if depth_conv:
        return dc.DepthConv(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False,
                            dilation=dilation, alpha=depth_conv_alpha)
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                     bias=False, dilation=dilation)


class AlexNet(nn.Module):

    def __init__(self, num_classes: int = 1000, in_channels: int = 3, depth_conv_alpha: float = 8.3,
                 depth_convs: Tuple[bool, bool, bool, bool, bool] = None) -> None:
        super(AlexNet, self).__init__()
        self.has_depth_conv = any(depth_convs)
        self.conv1 = conv(in_channels, 64, kernel_size=11, stride=4, padding=2, depth_conv=depth_convs[0],
                          depth_conv_alpha=depth_conv_alpha)
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool_2d_1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = conv(64, 192, kernel_size=5, padding=2, depth_conv=depth_convs[1],
                          depth_conv_alpha=depth_conv_alpha)
        self.relu2 = nn.ReLU(inplace=True)
        self.max_pool_2d_2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = conv(192, 384, kernel_size=3, padding=1, depth_conv=depth_convs[2],
                          depth_conv_alpha=depth_conv_alpha)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = conv(384, 256, kernel_size=3, padding=1, depth_conv=depth_convs[3],
                          depth_conv_alpha=depth_conv_alpha)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = conv(256, 256, kernel_size=3, padding=1, depth_conv=depth_convs[4],
                          depth_conv_alpha=depth_conv_alpha)
        self.relu5 = nn.ReLU(inplace=True)
        self.max_pool_2d_3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.depth_down_sampler = nn.AvgPool2d(3, stride=2) if self.has_depth_conv else None

    def forward(self, x: torch.Tensor, depth: Optional[torch.Tensor] = None) -> torch.Tensor:
        #x, depth = mu.sep_rgbd_data(x, self.has_depth_conv)
        x, depth = mu.forward_conv(x, depth, self.conv1, self.has_depth_conv, self.depth_down_sampler)
        x = self.relu1(x)
        x = self.max_pool_2d_1(x)
        x, depth = mu.forward_conv(x, depth, self.conv2, self.has_depth_conv, self.depth_down_sampler)
        x = self.relu2(x)
        x = self.max_pool_2d_2(x)
        x, depth = mu.forward_conv(x, depth, self.conv3, self.has_depth_conv, self.depth_down_sampler)
        x = self.relu3(x)
        x, depth = mu.forward_conv(x, depth, self.conv4, self.has_depth_conv, self.depth_down_sampler)
        x = self.relu4(x)
        x, depth = mu.forward_conv(x, depth, self.conv5, self.has_depth_conv, self.depth_down_sampler)
        x = self.relu5(x)
        x = self.max_pool_2d_3(x)
        return {'out': x}  # change me


def alexnet(pretrained: bool = False, progress: bool = True, in_channels: int = 3,
            depth_conv_option: Optional[str] = False, depth_conv_alpha: float = 8.3, **kwargs: Any) -> AlexNet:
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if depth_conv_option is None:
        depth_convs = (False, False, False, False, False)
    elif depth_conv_option == 'all':
        depth_convs = (True, True, True, True, True)
    elif depth_conv_option == 'front':
        depth_convs = (True, True, False, False, False)
    elif depth_conv_option == 'back':
        depth_convs = (False, False, False, True, True)
    else:
        raise ValueError(f'{depth_conv_option} is not a supported option')
    model = AlexNet(in_channels=in_channels, depth_convs=depth_convs, depth_conv_alpha=depth_conv_alpha, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model
