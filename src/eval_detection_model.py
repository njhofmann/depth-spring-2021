import torch.nn as nn
import torch.utils.data as d
import podm.podm as p
import itertools as it


def eval_detect_model(model: nn.Module, test_data: d.DataLoader, device):
    true_boxes, pred_boxes = [], []
    for i, (channels, bounding_boxes, box_classes) in enumerate(test_data):
        channels, bounding_boxes, box_classes = channels.to(device), bounding_boxes.to(device), box_classes.to(device)
        # pred_seg_mask = t.argmax(t.softmax(model(channels)['out'], dim=1), dim=1)
        true_boxes.append((bounding_boxes, box_classes))

        prediction = model(channels)
        pred_boxes.append((prediction['boxes'], prediction['labels']))
    # TODO class names
    true_boxes = [p.BoundingBox('foo', class_id, *box) for box, class_id in true_boxes]
    pred_boxes = [p.BoundingBox('foo', class_id, *box) for box, class_id in pred_boxes]
    return p.get_pascal_voc_metrics(true_boxes, pred_boxes)
