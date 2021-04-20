from typing import Tuple, List
from collections import OrderedDict
import torch
import warnings

from torchvision.models import detection as td

"""Overloaded version of PyTorch's FasterRCNN implementation that is split across two GPUs"""


class MultiGPUFasterRCNN(td.FasterRCNN):
    def __init__(self, backbone, num_classes=None,
                 # transform parameters
                 min_size=800, max_size=1333,
                 image_mean=None, image_std=None,
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 rpn_score_thresh=0.0,
                 # Box parameters
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512, box_positive_fraction=0.25,
                 bbox_reg_weights=None):
        super(MultiGPUFasterRCNN, self).__init__(backbone=backbone, num_classes=num_classes,
                                                 # transform parameters
                                                 min_size=min_size, max_size=max_size,
                                                 image_mean=image_mean, image_std=image_std,
                                                 # RPN parameters
                                                 rpn_anchor_generator=rpn_anchor_generator, rpn_head=rpn_head,
                                                 rpn_pre_nms_top_n_train=rpn_pre_nms_top_n_train,
                                                 rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
                                                 rpn_post_nms_top_n_train=rpn_post_nms_top_n_train,
                                                 rpn_post_nms_top_n_test=rpn_post_nms_top_n_test,
                                                 rpn_nms_thresh=rpn_nms_thresh,
                                                 rpn_fg_iou_thresh=rpn_fg_iou_thresh,
                                                 rpn_bg_iou_thresh=rpn_bg_iou_thresh,
                                                 rpn_batch_size_per_image=rpn_batch_size_per_image,
                                                 rpn_positive_fraction=rpn_positive_fraction,
                                                 rpn_score_thresh=rpn_score_thresh,
                                                 # Box parameters
                                                 box_roi_pool=box_roi_pool, box_head=box_head,
                                                 box_predictor=box_predictor,
                                                 box_score_thresh=box_score_thresh, box_nms_thresh=box_nms_thresh,
                                                 box_detections_per_img=box_detections_per_img,
                                                 box_fg_iou_thresh=box_fg_iou_thresh,
                                                 box_bg_iou_thresh=box_bg_iou_thresh,
                                                 box_batch_size_per_image=box_batch_size_per_image,
                                                 box_positive_fraction=box_positive_fraction,
                                                 bbox_reg_weights=bbox_reg_weights)

        self.transform = self.transform('cuda:0')
        self.backbone = self.backbone.to('cuda:0')
        self.rpn = self.rpn.to('cuda:1')
        self.roi_heads = self.roi_heads('cuda:1')

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)
        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                         "of shape [N, 4], got {:}.".format(
                                             boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError("All bounding boxes should have positive height and width."
                                     " Found invalid box {} for target at index {}."
                                     .format(degen_bb, target_idx))

        features = self.backbone(images.tensors).to('cuda:1')
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        else:
            return self.eager_outputs(losses, detections)