import abc
import pathlib as pl
from typing import Set, Tuple, Callable
import random as r

from PIL import Image as pi, ImageDraw as pid
import matplotlib.pyplot as plt
import numpy as np
import torch as t
import torch.utils.data as tud
import torchvision.transforms as tvt
import torchvision.transforms.functional as tvf
import albumentations.augmentations.functional as af

import paths as p

INPUT_SHAPE = (425, 560)
MAX_DEPTH = 65400  # from training data
CHANNEL_MEANS = t.tensor([0.4902, 0.4525, 0.4251, 0.2519])
CHANNEL_STDS = t.tensor([0.2512, 0.2564, 0.2581, 0.1227])


def get_unique_semantic_labels() -> Set[int]:
    """Utility function to check the integers making up semantic images"""
    idxs = set()
    data = SUNRGBDTrainDataset(True)
    for i in range(len(data)):
        idxs.update([x.item() for x in t.unique(data[i][1])])
    return idxs


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


class GenericSUNRGBDDataset(tud.Dataset, abc.ABC):
    CLASS_COUNT = 38

    def __init__(self, dirc: pl.Path, semantic_or_box: bool, rgb: bool, depth: bool, augment: bool = True) -> None:
        self.dircs = list(dirc.glob('*'))
        self.dircs.sort()

        # load semantic segmentation masks or 2D bounding boxes
        self.semantic_or_box = semantic_or_box

        if not (rgb or depth):
            raise ValueError('need to load at one type of image data, rgb or depth')

        # to augment the raw data or not
        self.augment = augment

        # to normalize images or not, only for testing purposes
        self.normalize = True

        # to load only rgb images only
        self.include_rgb = rgb

        # to load only depth info only
        self.include_depth = depth

        self.tensorer = tvt.ToTensor()
        self.rgb_and_depth_normal = tvt.Normalize(CHANNEL_MEANS, CHANNEL_STDS)
        self.rgb_normal = tvt.Normalize(CHANNEL_MEANS[:3], CHANNEL_STDS[:3])
        self.depth_normal = tvt.Normalize(CHANNEL_MEANS[-1], CHANNEL_STDS[-1])

    def _load_label(self, dirc: pl.Path, semantic: bool):
        return (self._load_semantic_label if semantic else self._load_bounding_boxes)(dirc)

    def _load_bounding_boxes(self, dirc: pl.Path):
        bounding_boxes = np.load(dirc.joinpath('bounding_box.npy'), allow_pickle=True)
        boxes = []
        for box in bounding_boxes[0]['gtBb2D']:
            box = [max(0, round(x)) for x in box[0]]
            box[2] += box[0]
            box[3] += box[1]
            boxes.append(box)
        return boxes

    def _load_semantic_label(self, dirc: pl.Path):
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
        return self.dircs[idx]

    def __getitem__(self, idx: int):
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

        return img, label

    def __len__(self):
        return len(self.dircs)

    def _apply_bbox_transform(self, boxes: list, transform: Callable, *args):
        return [transform(box, *args) for box in boxes]

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
            box_draw.rectangle(box, outline='black', width=5)
        return np.array(box_img) / 255

    def _channels_first(self, img: t.Tensor) -> np.ndarray:
        return np.transpose(img, (1, 2, 0))

    def view_raw_img(self, idx: int) -> None:
        """Utility method for viewing an image, its depth map, its semantic segmentation labels, and its bounding boxes
        all at once"""
        sample_dirc = self._get_sample_dirc(idx)
        rgb = self._load_rgb(sample_dirc)
        depth = self._load_depth(sample_dirc)
        boxes = self._load_label(sample_dirc, False)
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
        img = img[:3]

        if self.semantic_or_box:
            label_cmap = 'flag'
            # TODO fix me, add additional bounding box
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

    def __init__(self, semantic_or_box: bool, rgb: bool = True, depth: bool = True, augment: bool = True):
        super().__init__(p.SUN_RGBD_TRAIN_DIRC, semantic_or_box, rgb, depth, augment)
        self.cropper = tvt.RandomCrop(INPUT_SHAPE)
        self.jitter = tvt.ColorJitter()

    def _random_select(self, prob: float = .5) -> float:
        return r.random() < prob

    def _flip_img(self, img, label):
        if self._random_select():
            img = tvf.vflip(img)

            if self.semantic_or_box:
                label = tvf.vflip(label)
            else:
                rows, cols = img.shape[-2:]
                print(label)
                label = super()._apply_bbox_transform(label, af.bbox_vflip, rows, cols)
                print(label)
                label = [tuple(abs(i) for i in x) for x in label]

        if self._random_select():
            img = tvf.hflip(img)

            if self.semantic_or_box:
                label = tvf.hflip(label)
            else:
                rows, cols = img.shape[-2:]
                # TODO fix this, negative coordinates given
                print(label)
                label = super()._apply_bbox_transform(label, af.bbox_hflip, rows, cols)
                print(label)
                label = [tuple(abs(i) for i in x) for x in label]

        return img, label

    def _rotate_img(self, img, label):
        if self._random_select(.25):
            img = tvf.rotate(img, 180)

            if self.semantic_or_box:
                label = tvf.rotate(label, 180)
            else:
                rows, cols = img.shape[-2:]
                label = super()._apply_bbox_transform(label, af.bbox_rot90, 2, rows, cols)
                print(label)
        return img, label

    def _crop_img(self, img, label):
        cropper_params = self.cropper.get_params(img, INPUT_SHAPE)
        img = tvf.crop(img, *cropper_params)

        if self.semantic_or_box:
            label = tvf.crop(label, *cropper_params)
        else:
            rows, cols = img.shape[-2:]
            label = super()._apply_bbox_transform(label, af.bbox_crop, *cropper_params, rows, cols)

        return img, label

    def _jitter_img(self, img, label):
        if self.include_rgb and self._random_select(.3):
            img[:3] = self.jitter(img[:3])
        return img, label

    def _apply_augments(self, img, label):
        # TODO for bounding box
        img, label = super()._apply_augments(img, label)

        if not self.augment:
            return img, label

        # keep this this order
        # TODO bounding box jittering...?
        for func in self._crop_img, self._jitter_img, self._flip_img, self._rotate_img:
            img, label = func(img, label)
        return img, label


class SUNRGBDTestDataset(GenericSUNRGBDDataset):

    def __init__(self, semantic_or_box: bool, rgb: bool = True, depth: bool = True):
        super().__init__(p.SUN_RGBD_TEST_DIRC, semantic_or_box, rgb, depth)
        self.cropper = tvt.CenterCrop(INPUT_SHAPE)

    def _crop_img(self, img, label):
        img = self.cropper(img)
        if self.semantic_or_box:
            label = self.cropper(label)
        else:
            rows, cols = img.shape[-2:]
            # TODO fix me
            label = super()._apply_bbox_transform(label, af.bbox_center_crop, *INPUT_SHAPE, rows, cols)
        return img, label

    def _apply_augments(self, img, label):
        img, label = super()._apply_augments(img, label)
        img, label = self._crop_img(img, label)
        return img, label


def load_sun_rgbd_dataset(semantic_or_box: bool, include_rgb: bool, include_depth: bool, augment: bool) \
        -> Tuple[SUNRGBDTrainDataset, SUNRGBDTestDataset]:
    return SUNRGBDTrainDataset(semantic_or_box, include_rgb, include_depth, augment), \
           SUNRGBDTestDataset(semantic_or_box, include_rgb, include_depth)


if __name__ == '__main__':
    a = SUNRGBDTestDataset(False)
    a.view_raw_img(0)
