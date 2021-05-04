import abc
import pathlib as pl
from typing import Tuple, Union, List, Optional
import random as r
import collections as c

from PIL import Image as pi, ImageDraw as pid
import matplotlib.pyplot as plt
import numpy as np
import torch as t
import torch.utils.data as tud
import torchvision.transforms as tvt
import torchvision.transforms.functional as tvf

import paths as p
from src.datasets import custom_center_crop as cc, bbox_augments as bb

INPUT_SHAPE = (425, 560)
MAX_DEPTH = 65400  # from training data
CHANNEL_MEANS = t.tensor([0.4902, 0.4525, 0.4251, 0.2519])
CHANNEL_STDS = t.tensor([0.2512, 0.2564, 0.2581, 0.1227])


class GenericSUNRGBDDataset(tud.Dataset, abc.ABC):
    CLASS_COUNT = 38

    def __init__(self, dirc: pl.Path, semantic_or_box: bool, rgb: bool, depth: bool, augment: bool = True,
                 sep_rgbd: bool = False, bbox_cls_threshold: int = 300, bbox_classes: Optional[List[str]] = None) \
            -> None:
        # TODO explain me

        if not (rgb or depth):
            raise ValueError('need to load at one type of image data, rgb or depth')

        self.dircs = list(dirc.glob('*'))
        self.dircs.sort()

        # load semantic segmentation masks or 2D bounding boxes
        self.semantic_or_box = semantic_or_box

        # if both rgb and depth data, keep as separate inputs or a single input
        self.sep_rgbd = sep_rgbd

        # to augment the raw data or not
        self.augment = augment

        # to normalize images or not, only for testing purposes
        self.normalize = True

        # to load rgb info
        self.include_rgb = rgb

        # to load depth info
        self.include_depth = depth

        # indices for samples that contain the supported bbox classes
        self.bbox_indces, self.bbox_classes = None, None
        if not self.semantic_or_box:
            self.bbox_indces, self.bbox_classes = self._set_bbox_info(bbox_cls_threshold, bbox_classes)

        self.tensorer = tvt.ToTensor()  # TODO get rid of me
        self.rgb_and_depth_normal = tvt.Normalize(CHANNEL_MEANS, CHANNEL_STDS)
        self.rgb_normal = tvt.Normalize(CHANNEL_MEANS[:3], CHANNEL_STDS[:3])
        self.depth_normal = tvt.Normalize(CHANNEL_MEANS[-1], CHANNEL_STDS[-1])

    def _set_bbox_info(self, bbox_class_threshold: int, bbox_classes: Optional[List[int]] = None) \
            -> Tuple[List[int], List[str]]:
        # TODO explain me
        if bbox_classes is None:
            ids_to_count = c.Counter()
            for i in range(len(self.dircs)):
                bbox_info = self._load_bounding_box_info(self.dircs[i])
                if len(bbox_info) > 0:
                    labels = [x[0] for x in bbox_info[0]['classname']]
                    if len(labels) > 0:
                        for idx, x in enumerate(labels):
                            ids_to_count[x] += 1

            remain_counts = {class_id: count for class_id, count in ids_to_count.items() if count > bbox_class_threshold}

            print(f'{len(remain_counts)} classes remain, '
                  f'{sum(remain_counts.values()) / sum(ids_to_count.values())}% bounding boxes remain')
            print(list(sorted(remain_counts, key=lambda x: x[1])))

            bbox_classes = remain_counts.keys()

        non_empty_class_idxs = []
        for i in range(len(self.dircs)):
            bbox_info = self._load_bounding_box_info(self.dircs[i])
            if len(bbox_info) > 0:
                labels = [x[0] for x in bbox_info[0]['classname']]
                filtered_labels = [x for x in labels if x in bbox_classes]
                if len(filtered_labels) > 0:
                    non_empty_class_idxs.append(i)

        return non_empty_class_idxs, list(sorted(bbox_classes))

    def _load_label(self, dirc: pl.Path, semantic: bool) -> Union[t.Tensor, Tuple[t.Tensor, t.Tensor]]:
        return (self._load_semantic_label if semantic else self._load_bounding_boxes)(dirc)

    def _load_bounding_box_info(self, dirc: pl.Path) -> np.ndarray:
        return np.load(dirc.joinpath('bounding_box.npy'), allow_pickle=True)

    def _load_bounding_boxes(self, dirc: pl.Path) -> Tuple[t.Tensor, t.Tensor]:
        # TODO explain me
        bbox_info = self._load_bounding_box_info(dirc)
        raw_bboxes = [x for x in bbox_info[0]['gtBb2D']]
        raw_bbox_labels = [x[0] for x in bbox_info[0]['classname']]

        bboxes, bbox_labels = [], []
        for idx, box in enumerate(raw_bboxes):
            cur_raw_label = raw_bbox_labels[idx]
            if cur_raw_label in self.bbox_classes:
                box = [max(0, round(x)) for x in box[0]]
                bboxes.append([box[0], box[1], box[2] + box[0], box[3] + box[1]])
                bbox_labels.append(self.bbox_classes.index(cur_raw_label))

        return t.as_tensor(bboxes), t.as_tensor(bbox_labels)

    def _load_semantic_label(self, dirc: pl.Path) -> t.Tensor:
        return t.as_tensor(self.tensorer(pi.open(dirc.joinpath('semantic_segs.png'))) * 255, dtype=t.long)

    def _load_rgb(self, dirc: pl.Path) -> t.Tensor:
        return self.tensorer(np.array(pi.open(dirc.joinpath('rgb.png'))))

    def _load_depth(self, dirc: pl.Path) -> t.Tensor:
        return self.tensorer(np.array(pi.open(dirc.joinpath('depth.png')))).to(t.float) / MAX_DEPTH

    @abc.abstractmethod
    def _apply_augments(self, img, label):
        if not self.normalize:
            return img, label

        normalize = self.depth_normal
        if self.include_rgb and self.include_depth:
            normalize = self.rgb_and_depth_normal
        elif self.include_rgb:
            normalize = self.rgb_normal
        img = normalize(img)
        return img, label

    def _get_sample_dirc(self, idx: int) -> pl.Path:
        # explain me
        if not self.semantic_or_box:
            idx = self.bbox_indces[idx]
        return self.dircs[idx]

    def __getitem__(self, idx: int) -> Union[Tuple[t.Tensor, t.Tensor, t.Tensor], Tuple[t.Tensor, t.Tensor]]:
        sample_dirc = self._get_sample_dirc(idx)
        rgb_img, depth_img = None, None

        if self.include_rgb:
            rgb_img = self._load_rgb(sample_dirc)

        if self.include_depth:
            depth_img = self._load_depth(sample_dirc)

        if self.include_rgb and self.include_depth:
            img = t.cat((rgb_img, depth_img), 0)
        elif self.include_rgb:
            img = rgb_img
        else:
            img = depth_img

        label = self._load_label(sample_dirc, self.semantic_or_box)

        # sanity check
        if self.semantic_or_box and (a := img.shape[-2:]) != (b := label.shape[-2:]):
            raise ValueError(f'image and semantic mask have different sizes: {a} vs {b}')

        img, label = self._apply_augments(img, label)

        if self.semantic_or_box:
            label = label[0]

        if self.semantic_or_box:
            if self.sep_rgbd:
                return *self._sep_rgbd_data(img), label
            return img, label

        bboxes = t.Tensor(label[0])
        bbox_labels = t.LongTensor(label[1])
        if len(bboxes) != len(bbox_labels):
            raise ValueError(f'{len(label)} {len(bbox_labels)}')
        return img, bboxes, bbox_labels

    def __len__(self):
        return len(self.dircs if self.semantic_or_box else self.bbox_indces)

    def _sep_rgbd_data(self, rgbd: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        depth = rgbd[:, 3, :, :]
        depth = depth[:, None, :, :]  # [batch_sz, h, w] --> [batch_sz, dummy_dim, h, w]
        rgb = rgbd[:, 0:3, :, :]
        return rgb, depth

    @property
    def channel_count(self) -> int:
        cnt = 0
        if self.include_rgb:
            cnt += 3
        if self.include_depth:
            cnt += 1
        return cnt

    def _draw_boxes(self, rgb_img: t.Tensor, boxes: t.Tensor) -> np.ndarray:
        box_img = tvt.ToPILImage(mode='RGB')(rgb_img.clone())
        box_draw = pid.Draw(box_img)
        for box in boxes:
            if any([x < 0 for x in box]):
                raise ValueError(f'negative coordinate found in bounding box {box}')
            box_draw.rectangle(list(box), outline='black', width=5)
        return np.array(box_img) / 255

    def _channels_first(self, img: t.Tensor) -> np.ndarray:
        return np.transpose(img, (1, 2, 0))

    def view_raw_img(self, idx: int) -> None:
        """Utility method for viewing an image, its depth map, its semantic segmentation labels, and its bounding boxes
        all at once"""
        sample_dirc = self._get_sample_dirc(idx)
        rgb = self._load_rgb(sample_dirc)
        depth = self._load_depth(sample_dirc)
        boxes = self._load_label(sample_dirc, False)[0]
        sem_seg = self._load_label(sample_dirc, True)

        box_img = self._draw_boxes(rgb, boxes)

        rgb, depth, sem_seg = [self._channels_first(x) for x in (rgb, depth, sem_seg)]

        fig = plt.figure()
        rows, cols = 2, 2
        i = 1
        for img, cmap in (rgb, None), (depth, 'gray'), (box_img, None), (sem_seg, 'flag'):
            fig.add_subplot(rows, cols, i)
            plt.imshow(img, cmap=cmap)
            i += 1
        plt.show()

    def view_img(self, idx):
        self.normalize = False
        img, label = self[idx]

        if not self.semantic_or_box:
            label = label[0]

        img = img[:3]

        if self.semantic_or_box:
            label_cmap = 'flag'
            label = self._channels_first(label)
        else:
            label_cmap = None
            label = self._draw_boxes(img, label)

        img = self._channels_first(img)

        rows, cols = 1, 2
        fig = plt.figure()
        for i, (x, cmap) in enumerate([(img, None), (label, label_cmap)]):
            fig.add_subplot(rows, cols, i + 1)
            plt.imshow(x, cmap=cmap)
        plt.show()


class SUNRGBDTrainDataset(GenericSUNRGBDDataset):

    def __init__(self, semantic_or_box: bool, rgb: bool = True, depth: bool = True, augment: bool = True,
                 sep_rgbd: bool = False) -> None:
        super().__init__(p.SUN_RGBD_TRAIN_DIRC, semantic_or_box, rgb, depth, augment, sep_rgbd=sep_rgbd)
        self.cropper = tvt.RandomCrop(INPUT_SHAPE)
        self.jitter = tvt.ColorJitter()

    def _random_select(self, prob: float = .5) -> float:
        return r.random() < prob

    def _flip_img(self, img, label):
        # double flip == rotation by 180
        if self._random_select():
            img = tvf.vflip(img)

            if self.semantic_or_box:
                label = tvf.vflip(label)
            else:
                bboxes, labels = label
                bboxes = bb.bbox_vflip(bboxes, img.shape[-2])
                label = bboxes, labels

        if self._random_select():
            img = tvf.hflip(img)

            if self.semantic_or_box:
                label = tvf.hflip(label)
            else:
                bboxes, labels = label
                bboxes = bb.bbox_hflip(bboxes, img.shape[-1])
                label = bboxes, labels

        return img, label

    def _crop_img(self, img, label, cropper_params: Optional[Tuple[int, int, int, int]] = None):
        # TODO abstract this out
        if cropper_params is None:
            cropper_params = self.cropper.get_params(img, INPUT_SHAPE)

        new_img = tvf.crop(img, *cropper_params)
        if self.semantic_or_box:
            new_label = tvf.crop(label, *cropper_params)
        else:
            cropper_params = bb.adjust_torch_crop_params(cropper_params)
            new_label = bb.bbox_crop(*label, *cropper_params)

            # if crop results in no bounding boxes, try again
            # TODO move this to a utility checker function before any actual cropping
            if len(new_label[0]) == 0:
                return self._crop_img(img, label)

        return new_img, new_label

    def _jitter_img(self, img, label):
        if self.include_rgb and self._random_select(.5):
            img[:3] = self.jitter(img[:3])
        return img, label

    def _apply_augments(self, img, label):
        img, label = super()._apply_augments(img, label)

        if not self.augment:
            return img, label

        # keep this this order
        # TODO bounding box jittering...?
        for func in self._jitter_img, self._flip_img, self._crop_img:
            img, label = func(img, label)
        return img, label


class SUNRGBDTestDataset(GenericSUNRGBDDataset):

    def __init__(self, semantic_or_box: bool, rgb: bool = True, depth: bool = True, sep_rgbd: bool = False,
                 bbox_classes: Optional[List[str]] = None) -> None:
        super().__init__(p.SUN_RGBD_TEST_DIRC, semantic_or_box, rgb, depth, sep_rgbd=sep_rgbd,
                         bbox_classes=bbox_classes)

    def _crop_img(self, img, label):
        cropper_params = cc.center_crop(img, list(INPUT_SHAPE))
        img = tvf.crop(img, *cropper_params)
        if self.semantic_or_box:
            label = tvf.crop(label, *cropper_params)
        else:
            cropper_params = bb.adjust_torch_crop_params(cropper_params)
            label = bb.bbox_crop(*label, *cropper_params)
        return img, label

    def _apply_augments(self, img, label):
        img, label = super()._apply_augments(img, label)
        img, label = self._crop_img(img, label)
        return img, label


def load_sun_rgbd_dataset(segmentation_or_box: bool, include_rgb: bool, include_depth: bool, augment: bool,
                          sep_rgbd: bool) -> Tuple[SUNRGBDTrainDataset, SUNRGBDTestDataset]:
    train = SUNRGBDTrainDataset(segmentation_or_box, include_rgb, include_depth, augment, sep_rgbd=sep_rgbd)
    test = SUNRGBDTestDataset(segmentation_or_box, include_rgb, include_depth, sep_rgbd=sep_rgbd,
                              bbox_classes=train.bbox_classes)

    if not segmentation_or_box and train.bbox_classes != test.bbox_classes:
        raise RuntimeError('train and test classes have different bbox classes')

    return train, test


if __name__ == '__main__':
    a = SUNRGBDTrainDataset(True)
    print(a[0][0].shape)