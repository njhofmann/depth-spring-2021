import torchvision.models.segmentation._utils as su
from typing import Optional
from collections import OrderedDict
import torch.nn.functional as F
import torch as t


class _SimpleDepthSegmentationmodel(su._SimpleSegmentationModel):
    """Overridden version of PyTorch's DeepLabV3 implementation that has modified the forward call so the backbone
    model may take in an optional depth input

    From: https://github.com/pytorch/vision/blob/master/torchvision/models/segmentation/_utils.py
    """

    def __init__(self, backbone, classifier):
        super(_SimpleDepthSegmentationmodel, self).__init__(backbone, classifier)

    def forward(self, x: t.Tensor, depth: Optional[t.Tensor] = None):
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x, depth)

        result = OrderedDict()
        x = features["out"]
        x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        result["out"] = x

        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            result["aux"] = x

        return result


class DepthDeepLabV3(_SimpleDepthSegmentationmodel):
    """Version of PyTorch's DeepLabv3 implementation where the backbone model can take in an optional depth input"""
    pass
