import torch.utils.data as tud
import paths as p
import torch as t
import numpy as np
import pathlib as pl
import torchvision.transforms as tvt


def load_img_as_tensor(path):
    return t.from_numpy(np.load(path))


class GenericSUNRGBDDataset(tud.Dataset):

    def __init__(self, dirc: pl.Path, semantic_or_box: bool, rgb: bool, depth: bool):
        self.dircs = list(dirc.glob('*'))
        self.dircs.sort()

        # load semantic segmentation masks or 2D bounding boxes
        self.semantic_or_box = semantic_or_box

        if not (rgb or depth):
            raise ValueError('need to load at one type of image data')

        # to load only rgb images only
        self.rgb_only = rgb

        # to load only depth info only
        self.depth_only = depth

        self.resizer = tvt.Resize((640, 480))

    def __getitem__(self, idx):
        # TODO figure out proper formatting for this
        item = self.dircs[idx]
        if self.rgb_only and self.depth_only:
            rgb_img = load_img_as_tensor(item.joinpath('rgb.npy'))
            depth_img = load_img_as_tensor(item.joinpath('depth.npy'))
            img = t.cat((rgb_img, depth_img), 0)
        elif self.rgb_only:
            img = load_img_as_tensor(item.joinpath('rgb.npy'))
        else:  # should only be depth image
            img = np.load(item.joinpath('depth.npy'))

        img = self.resizer.forward(img)
        label = np.load(item.joinpath('semantic_seg.npy' if self.semantic_or_box else 'bounding_box.npy'))
        return img, label

    def __len__(self):
        return len(self.dircs)


class SUNRGBDTrainDataset(GenericSUNRGBDDataset):

    def __init__(self, semantic_or_box: bool, rgb: bool = False, depth: bool = False):
        super().__init__(p.SUN_RGBD_TRAIN_DIRC, semantic_or_box, rgb, depth)


class SUNRGBDTestDataset(GenericSUNRGBDDataset):

    def __init__(self, semantic_or_box: bool, rgb: bool = False, depth: bool = False):
        super().__init__(p.SUN_RGBD_TEST_DIRC, semantic_or_box, rgb, depth)
