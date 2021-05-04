try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

import torch as t
from typing import Tuple, Optional, Union
import torch.nn as nn
import depth_conv_ops.depthaware.models.ops.depthconv.module as dc


def sep_rgbd_data(x: t.Tensor, has_depth_conv: bool) -> Tuple[t.Tensor, Optional[t.Tensor]]:
    # if using depth convs, separate info
    depth = None
    if has_depth_conv and x.shape[1] == 4:
        depth = x[:, 3, :, :]
        depth = depth[:, None, :, :]  # [batch_sz, h, w] --> [batch_sz, dummy_dim, h, w]
        x = x[:, 0:3, :, :]
        depth = depth.contiguous()
    x = x.contiguous()
    return x, depth


def forward_conv(x: t.Tensor, depth: Optional[t.Tensor], conv: Union[dc.DepthConv, nn.Conv2d],
                 has_depth_conv: bool, depth_avger: nn.AvgPool2d) -> Tuple[t.Tensor, Optional[t.Tensor]]:
    # downscale depth until same shape
    while has_depth_conv and depth.shape[-2:] != x.shape[-2:]:
        depth = depth_avger(depth)

    if isinstance(conv, dc.DepthConv) and has_depth_conv and depth is not None:
        return conv(x, depth), depth
    return conv(x), depth
