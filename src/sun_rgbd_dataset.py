import abc
import pathlib as pl
from typing import Set, Tuple
import random as r

import PIL.Image as pi
import numpy as np
import torch as t
import torch.utils.data as tud
import torchvision.transforms as tvt
import torchvision.transforms.functional as tvf

import paths as p

# INPUT_SIZE = (640, 480)  # around a 4 / 3 ratio
INPUT_SIZE = 425
MAX_DEPTH = 65400  # from training data
CHANNEL_MEANS = t.tensor([0.4902, 0.4525, 0.4251, 0.2519])
CHANNEL_STDS = t.tensor([0.2512, 0.2564, 0.2581, 0.1227])


def get_unique_semantic_labels() -> Set[int]:
    """Utility function to check the integers making up semantic images"""
    idxs = set()
    data = SUNRGBDTrainDataset(True)
    for i in range(len(data)):
        print(i)
        idxs.update([x.item() for x in t.unique(data[i][1])])
    return idxs


def get_max_depth_val():
    """Returns the largest depth value in the trainig dataset to use for scaling depth images [0, 1.0]. Computationally
    expensive, don't run unless you have to."""
    data = SUNRGBDTrainDataset(True)
    return max([data[0][i][-1].flatten().item() for i in range(len(data))])


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


class GenericSUNRGBDDataset(tud.Dataset):
    CLASS_COUNT = 38

    def __init__(self, dirc: pl.Path, semantic_or_box: bool, rgb: bool, depth: bool):
        self.dircs = list(dirc.glob('*'))
        self.dircs.sort()

        # load semantic segmentation masks or 2D bounding boxes
        self.semantic_or_box = semantic_or_box

        if not (rgb or depth):
            raise ValueError('need to load at one type of image data')

        # to load only rgb images only
        self.include_rgb = rgb

        # to load only depth info only
        self.include_depth = depth

        self.tensorer = tvt.ToTensor()
        self.rgb_and_depth_normal = tvt.Normalize(CHANNEL_MEANS, CHANNEL_STDS)
        self.rgb_normal = tvt.Normalize(CHANNEL_MEANS[:3], CHANNEL_STDS[:3])
        self.depth_normal = tvt.Normalize(CHANNEL_MEANS[-1], CHANNEL_STDS[-1])

    @abc.abstractmethod
    def _apply_augments(self, img, label):
        normalize = self.depth_normal
        if self.include_rgb and self.include_depth:
            normalize = self.rgb_and_depth_normal
        elif self.include_rgb:
            normalize = self.rgb_normal
        img = normalize(img)
        return img, label

    def __get_sample_dirc(self, idx: int) -> pl.Path:
        return self.dircs[idx]

    def __getitem__(self, idx: int):
        # TODO figure out proper formatting for this
        # TODO train vs test for transforms
        sample_dirc = self.__get_sample_dirc(idx)
        rgb_img, depth_img = None, None

        if self.include_rgb:
            rgb_img = self.tensorer(pi.open(sample_dirc.joinpath('rgb.png')))

        if self.include_depth:
            depth_img = self.tensorer(pi.open(sample_dirc.joinpath('depth.png'))).to(t.float) / MAX_DEPTH

        if self.include_rgb and self.include_depth:
            img = t.cat((rgb_img, depth_img), 0)
        elif self.include_rgb:
            img = rgb_img
        else:
            img = depth_img

        if self.semantic_or_box:
            label = t.as_tensor(self.tensorer(pi.open(sample_dirc.joinpath('semantic_segs.png')))[0] * 255,
                                dtype=t.long)
        else:
            label = np.load(sample_dirc.joinpath('bounding_box.npy'))

        # sanity check
        if (a := img.shape[-2:]) != (b := label.shape[-2:]):
            raise ValueError(f'image and semantic mask have different sizes: {a} vs {b}')

        return self._apply_augments(img, label)

    def __len__(self):
        return len(self.dircs)

    @property
    def channel_count(self) -> int:
        cnt = 0
        if self.include_rgb:
            cnt += 3
        if self.include_depth:
            cnt += 1
        return cnt

    def view_img(self, idx: int) -> None:
        """Utility method for viewing an image, its depth map, its semantic segmentation labels, and its bounding boxes
        all at once"""
        sample_dirc = self.__get_sample_dirc(idx)
        # TODO this


class SUNRGBDTrainDataset(GenericSUNRGBDDataset):
    TRANSFORM_PROB = .5

    def __init__(self, semantic_or_box: bool, rgb: bool = True, depth: bool = True):
        super().__init__(p.SUN_RGBD_TRAIN_DIRC, semantic_or_box, rgb, depth)
        self.cropper = tvt.RandomCrop(INPUT_SIZE)
        self.jitter = tvt.ColorJitter()

    def __random_select(self, prob: float = .5) -> float:
        return r.random() < prob

    def __flip_img(self, img, label):
        if self.__random_select():
            img = tvf.vflip(img)

            if self.semantic_or_box:
                label = tvf.vflip(label)

        if self.__random_select():
            img = tvf.hflip(img)

            if self.semantic_or_box:
                label = tvf.hflip(label)

        return img, label

    def __rotate_img(self, img, label):
        rotation = r.choices([0.0, 90.0, 180.0, 270.0], weights=[.5, .5 / 3, .5 / 3, .5 / 3], k=1)[0]
        img = tvf.rotate(img, rotation)
        if self.semantic_or_box:
            label = tvf.rotate(label, rotation)
        return img, label

    def _apply_augments(self, img, label):
        # TODO for bounding box
        img, label = super()._apply_augments(img, label)
        cropper_params = self.cropper.get_params(img, (INPUT_SIZE, INPUT_SIZE))
        img = tvf.crop(img, *cropper_params)

        if self.semantic_or_box:
            label = tvf.crop(label, *cropper_params)

        if self.include_rgb:
            img[:3] = self.jitter(img[:3])

        img, label = self.__flip_img(img, label)
        img, label = self.__rotate_img(img, label)
        return img, label


class SUNRGBDTestDataset(GenericSUNRGBDDataset):

    def __init__(self, semantic_or_box: bool, rgb: bool = True, depth: bool = True):
        super().__init__(p.SUN_RGBD_TEST_DIRC, semantic_or_box, rgb, depth)
        self.cropper = tvt.CenterCrop(INPUT_SIZE)

    def _apply_augments(self, img, label):
        # TODO for bounding box
        img, label = super()._apply_augments(img, label)
        img = self.cropper(img)
        if self.semantic_or_box:
            label = self.cropper(label)

        return img, label


def load_sun_rgbd_dataset(semantic_or_box: bool, include_rgb: bool, include_depth: bool) \
        -> Tuple[SUNRGBDTrainDataset, SUNRGBDTestDataset]:
    return SUNRGBDTrainDataset(semantic_or_box, include_rgb, include_depth), \
           SUNRGBDTestDataset(semantic_or_box, include_rgb, include_depth)


if __name__ == '__main__':
    data = SUNRGBDTrainDataset(True)
    for i in range(len(data)):
        a, b = data[i]
        print(a, b.shape)
