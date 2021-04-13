import collections as c
from typing import Set

import torch as t

from src.datasets.sun_rgbd_dataset import SUNRGBDTrainDataset, SUNRGBDTestDataset


def get_unique_semantic_labels() -> Set[int]:
    """Utility function to check the integers making up semantic images"""
    idxs = set()
    data = SUNRGBDTrainDataset(True)
    for i in range(len(data)):
        idxs.update([x.item() for x in t.unique(data[i][1])])
    return idxs


def get_bbox_label_info(class_threshold: int = 300) -> None:
    # TODO how many images are lost with this threshold
    ids_to_names = c.defaultdict(lambda: [0, set()])
    ids_to_count = c.Counter()
    data = SUNRGBDTrainDataset(False)
    for i in range(len(data)):
        try:
            bbox_info = data._load_bounding_box_info(data.dircs[i])
            labels = [x[0] for x in bbox_info[0]['classname']]
            bbox_ids = [x[0][0] for x in bbox_info[0]['objid']]

            for idx, x in enumerate(labels):
                ids_to_names[bbox_ids[idx]][0] += 1
                ids_to_names[bbox_ids[idx]][1].add(x)
                ids_to_count[x] += 1

        except IndexError:
            print(f'element {i} has no boxes')

    for k, (cnt, labels) in ids_to_names.items():
        print(k, cnt, len(labels), labels)

    remain_counts = list(filter(lambda x: x > class_threshold, ids_to_count.values()))
    print(f'{len(remain_counts)} classes remain, {sum(remain_counts) / sum(ids_to_count.values())}% bounding boxes remain')
    print(list(sorted(remain_counts)))


def get_max_depth_val():
    """Returns the largest depth value in the trainig dataset to use for scaling depth images [0, 1.0]. Computationally
    expensive, don't run unless you have to."""
    data = SUNRGBDTrainDataset(True)
    return max([data[0][i][-1].flatten().item() for i in range(len(data))])


def get_raw_image_sizes() -> set:
    """Dataset is a collection of several data sources, each with different sizes for their images. This method
    collects those sizes"""
    sizes = set()
    data = SUNRGBDTrainDataset(True, augment=False)
    for i in range(len(data)):
        sizes.add(data[i][0].shape)
    return sizes


def compute_training_stats():
    """Computes the mean and standard deviation for each RGB-D channel from the training data"""
    means, stds = [], []
    data = SUNRGBDTrainDataset(True)
    for i in range(len(data)):
        print(i)
        img, _ = data[i]
        std, mean = t.std_mean(input=img, dim=(1, 2))
        means.append(mean)
        stds.append(std)
    means = t.sum(t.vstack(means), dim=0) / len(means)
    stds = t.sum(t.vstack(stds), dim=0) / len(stds)
    print(means, stds)


if __name__ == '__main__':
    a = get_bbox_label_info()
