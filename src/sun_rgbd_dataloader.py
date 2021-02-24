import torch.utils.data as tud
import paths as p
import torch as t
import numpy as np
import pathlib as pl
import torchvision.transforms as tvt
import torchvision.transforms.functional as tvf
import PIL.Image as pi

# INPUT_SIZE = (640, 480)  # around a 4 / 3 ratio
INPUT_SIZE = 425
MAX_DEPTH = 65400  # from training data
CHANNEL_MEANS = t.tensor([0.4902, 0.4525, 0.4251, 0.2519])
CHANNEL_STDS = t.tensor([0.2512, 0.2564, 0.2581, 0.1227])


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


def load_img_as_tensor(path):
    return t.from_numpy(np.load(path))


class GenericSUNRGBDDataset(tud.Dataset):

    def __init__(self, dirc: pl.Path, semantic_or_box: bool, rgb: bool, depth: bool, train_or_test: bool):
        self.dircs = list(dirc.glob('*'))
        self.dircs.sort()

        # load semantic segmentation masks or 2D bounding boxes
        self.semantic_or_box = semantic_or_box

        self.train_or_test = train_or_test

        if not (rgb or depth):
            raise ValueError('need to load at one type of image data')

        # to load only rgb images only
        self.include_rgb = rgb

        # to load only depth info only
        self.include_depth = depth

        self.resizer = tvt.Resize(INPUT_SIZE)
        self.cropper = tvt.RandomCrop(INPUT_SIZE)
        self.common_transforms = tvt.Compose([self.resizer,
                                              tvt.ToTensor()])

        self.jitter = tvt.ColorJitter()
        self.tensorer = tvt.ToTensor()
        self.random_crop = tvt.RandomResizedCrop(100)
        self.depth_transforms = tvt.Compose([tvt.Resize(INPUT_SIZE),
                                             tvt.ToTensor()])

    def _apply_augments(self, rgb, depth, label):
        # TODO for bounding box
        cropper_params = self.cropper.get_params(rgb if self.include_rgb else depth, (INPUT_SIZE, INPUT_SIZE))
        if rgb is not None:
            rgb = tvf.crop(rgb, *cropper_params)
            rgb = self.jitter(rgb)

        if depth is not None:
            depth = tvf.crop(depth, *cropper_params)

        if label is not None:
            label = tvf.crop(label, *cropper_params)

        return rgb, depth, label

    def __getitem__(self, idx):
        # TODO figure out proper formatting for this
        # TODO train vs test for transforms
        item = self.dircs[idx]
        rgb_img, depth_img = None, None

        if self.include_rgb:
            rgb_img = self.tensorer(pi.open(item.joinpath('rgb.png')))
            for i in range(3):
                rgb_img[i] = (rgb_img[i] - CHANNEL_MEANS[i]) / CHANNEL_STDS[i]

        if self.include_depth:
            depth_img = self.tensorer(pi.open(item.joinpath('depth.png')))
            depth_img = depth_img.to(t.float)
            depth_img = ((depth_img / MAX_DEPTH) - CHANNEL_MEANS[-1]) / CHANNEL_STDS[-1]

        label = np.load(item.joinpath('semantic_segs.npy' if self.semantic_or_box else 'bounding_box.npy'))
        if self.semantic_or_box:
            label = self.resizer(tvt.ToTensor()(label))

        if self.train_or_test:
            # TODO if test, get consistent points
            rgb_img, depth_img, label = self._apply_augments(rgb_img, depth_img, label)

        if self.include_rgb and self.include_depth:
            img = t.cat((rgb_img, depth_img), 0)
        elif self.include_rgb:
            img = rgb_img
        else:
            img = depth_img
        return img, label

    def __len__(self):
        return len(self.dircs)


class SUNRGBDTrainDataset(GenericSUNRGBDDataset):

    def __init__(self, semantic_or_box: bool, rgb: bool = True, depth: bool = True):
        super().__init__(p.SUN_RGBD_TRAIN_DIRC, semantic_or_box, rgb, depth, True)


class SUNRGBDTestDataset(GenericSUNRGBDDataset):

    def __init__(self, semantic_or_box: bool, rgb: bool = True, depth: bool = True):
        super().__init__(p.SUN_RGBD_TEST_DIRC, semantic_or_box, rgb, depth, False)


if __name__ == '__main__':
    data = SUNRGBDTrainDataset(True)
    for i in range(len(data)):
        print(data[i])
