import torch.nn as nn
import torch.utils.data as d
import podm.podm as p
import itertools as it


def get_class_name(class_id: int, data_loader: d.DataLoader) -> str:
    return data_loader.dataset.bbox_classes[class_id]


def eval_detect_model(model: nn.Module, test_data: d.DataLoader, device):
    true_labels, pred_labels = [], []
    for i, (channels, bboxes, bbox_labels) in enumerate(test_data):
        # channels = channels.to(device)
        # for i in range(len(bboxes)):
        #     bboxes[i] = bboxes[i].to(device)
        #     bbox_labels[i] = bbox_labels[i].to(device)
        # pred_seg_mask = t.argmax(t.softmax(model(channels)['out'], dim=1), dim=1)
        true_labels.append((bboxes, bbox_labels))

        prediction = model(channels)
        pred_bboxes, pred_bbox_labels = [], []
        for i in range(len(prediction)):
            pred_bboxes.append(prediction[i]['boxes'])
            pred_bbox_labels.append((prediction[i]['labels']))

        pred_labels.append((pred_bboxes, pred_labels))

    true_labels = [p.BoundingBox(get_class_name(class_id, test_data), class_id, *box) for box, class_id in true_labels]
    pred_labels = [p.BoundingBox(get_class_name(class_id, test_data), class_id, *box) for box, class_id in pred_labels]
    return p.get_pascal_voc_metrics(true_labels, pred_labels)
