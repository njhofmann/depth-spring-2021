import torch
from torch import Tensor
import torch.nn as nn
from models.utils import load_state_dict_from_url
import models.utils as mu
from typing import Type, Any, Callable, Union, List, Optional, Tuple
import depth_conv_ops.depthaware.models.ops.depthconv.module as dc

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1,
            depth_conv_alpha: float = 8.3, depth_conv: bool = False) -> Union[nn.Conv2d, dc.DepthConv]:
    """3x3 convolution with padding"""

    if depth_conv:
        return dc.DepthConv(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, bias=False,
                            dilation=dilation, alpha=depth_conv_alpha)
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False,
                     dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, depth_conv_alpha: float = 8.3,
            depth_conv: bool = False, ) -> Union[dc.DepthConv, nn.Conv2d]:
    """1x1 convolution"""
    if depth_conv:
        return dc.DepthConv(in_planes, out_planes, alpha=depth_conv_alpha, kernel_size=1, stride=stride, bias=False)
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            depth_conv: bool = False,
            depth_conv_alpha: float = 8.3
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, depth_conv=depth_conv)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, depth_conv=depth_conv)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            depth_conv: bool = False,
            depth_conv_alpha: float = 8.3
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, depth_conv=depth_conv, depth_conv_alpha=depth_conv_alpha)
        self.conv1_depth_avger = self._init_depth_avger(depth_conv)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation, depth_conv=depth_conv,
                             depth_conv_alpha=depth_conv_alpha)
        self.conv2_depth_avger = self._init_depth_avger(depth_conv)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion, depth_conv=depth_conv,
                             depth_conv_alpha=depth_conv_alpha)
        self.conv3_depth_avger = self._init_depth_avger(depth_conv)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.depth_conv = depth_conv

    def _init_depth_avger(self, depth_conv: bool = False) -> Optional[nn.AvgPool2d]:
        if not depth_conv:
            return None
        return nn.AvgPool2d(3, padding=1, stride=2)

    def forward(self, x: Tuple[Tensor, Optional[Tensor]]) -> Tuple[Tensor, Optional[Tensor]]:
        x, depth = x
        identity = x

        out, _ = mu.forward_conv(x, depth, self.conv1, self.depth_conv, self.conv1_depth_avger)
        out = self.bn1(out)
        out = self.relu(out)

        out, _ = mu.forward_conv(out, depth, self.conv2, self.depth_conv, self.conv2_depth_avger)
        out = self.bn2(out)
        out = self.relu(out)

        out, _ = mu.forward_conv(out, depth, self.conv3, self.depth_conv, self.conv3_depth_avger)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out, depth


class ResNet(nn.Module):

    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[Union[str, int]],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            input_channels: int = 3,
            depth_conv_alpha: float = 8.3
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        layers, self.has_depth_conv = self._parse_depth_conv_options(layers)
        # treat first conv layer as apart of first bottleneck
        if layers[0][1]:
            self.conv1 = dc.DepthConv(input_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = nn.Conv2d(input_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0][0], depth_conv=layers[0][1])
        self.layer2 = self._make_layer(block, 128, layers[1][0], stride=2,
                                       dilate=replace_stride_with_dilation[0], depth_conv=layers[1][1],
                                       depth_conv_alpha=depth_conv_alpha)
        self.layer3 = self._make_layer(block, 256, layers[2][0], stride=2,
                                       dilate=replace_stride_with_dilation[1], depth_conv=layers[2][1],
                                       depth_conv_alpha=depth_conv_alpha)
        self.layer4 = self._make_layer(block, 512, layers[3][0], stride=2,
                                       dilate=replace_stride_with_dilation[2], depth_conv=layers[3][1],
                                       depth_conv_alpha=depth_conv_alpha)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _parse_depth_conv_options(self, layers: List[Union[int, str]]) -> Tuple[List[Tuple[int, bool]], bool]:
        # TODO explain me
        new_layers = []
        has_depth_conv = False
        for layer in layers:
            if isinstance(layer, int):
                new_layers.append((layer, False))
            elif isinstance(layer, str) and layer[-1] == 'C':
                new_layers.append((int(layer[:-1]), True))
                has_depth_conv = True
            else:
                raise ValueError(f'layer {layer} is malformed')
        return new_layers, has_depth_conv

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int, stride: int = 1,
                    dilate: bool = False, depth_conv: bool = False, depth_conv_alpha: float = 8.3) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),  # TODO make this depth conv?
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, depth_conv=depth_conv,
                            depth_conv_alpha=depth_conv_alpha))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, depth_conv=depth_conv,
                                depth_conv_alpha=depth_conv_alpha))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor, depth: Optional[Tensor] = None) -> Tensor:
        # See note [TorchScript super()]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x, _ = self.layer1((x, depth))
        x, _ = self.layer2((x, depth))
        x, _ = self.layer3((x, depth))
        x, _ = self.layer4((x, depth))

        return {'out': x}

    def forward(self, x: Tensor, depth: Optional[Tensor] = None) -> Tensor:
        return self._forward_impl(x, depth)


def _resnet(
        arch: str,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        pretrained: bool,
        progress: bool,
        input_channels: int = 3,
        depth_conv_alpha: float = 8.3,
        **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, input_channels=input_channels, depth_conv_alpha=depth_conv_alpha, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet50(pretrained: bool = False, progress: bool = True, depth_conv_alpha: float = 8.3, input_channels: int = 3,
             depth_conv_option: Optional[str] = None, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if depth_conv_option is None:
        layers = [3, 4, 6, 3]
    elif depth_conv_option == 'all':
        layers = ['3C', '4C', '6C', '3C']
    elif depth_conv_option == 'front':
        layers = ['3C', 4, 6, 3]
    elif depth_conv_option == 'back':
        layers = [3, 4, 6, '3C']
    else:
        raise ValueError(f'{depth_conv_option} is not a supported option')
    return _resnet('resnet50', Bottleneck, layers, pretrained, progress, input_channels, depth_conv_alpha,
                   **kwargs)
