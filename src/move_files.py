import pathlib as pl
from typing import Tuple, Set

import h5py as h
import numpy as np
import scipy.io as s

import paths as p

"""Script for parsing out SUN RGBD data into a more sane formatting"""


def load_seg_info() -> np.ndarray:
    with h.File(p.SUN_RGBD_DIRC.joinpath('SUNRGBDtoolbox/Metadata/SUNRGBD2Dseg.mat'), 'r') as f:
        segs_loc = f['SUNRGBD2Dseg']['seglabel']
        for i in range(segs_loc.shape[0]):
            yield np.array(f[segs_loc[i][0]])


def relink_image_path(old_path: str) -> pl.Path:
    new_path = p.SUN_RGBD_DIRC.joinpath('/'.join(old_path.split('/')[5:]))
    if not new_path.exists():
        raise ValueError(f'image path {new_path} doesn\'t exist')
    return new_path


def load_meta_info() -> Tuple[str, str, np.ndarray, np.ndarray]:
    mat_file = s.loadmat(p.SUN_RGBD_DIRC.joinpath('SUNRGBDtoolbox/Metadata/SUNRGBDMeta.mat'))
    paths = mat_file['SUNRGBDMeta']
    rgb_paths, depth_paths, bounding_boxes = paths['rgbpath'][0], paths['depthpath'][0], paths['groundtruth2DBB'][0]

    if len(rgb_paths) != len(depth_paths) != len(bounding_boxes):
        raise ValueError('number of RGB image paths, depth data paths, and bounding boxes must be the same')

    groups = [rgb_paths, depth_paths, bounding_boxes, load_seg_info()]
    for rgb_path, depth_path, bounding_box, seg_data in zip(*groups):
        yield rgb_path[0], depth_path[0], bounding_box, seg_data


def format_split_paths(paths: np.ndarray) -> Set[str]:
    return set(('/'.join(path[0].split('/')[5:]) for path in paths))


def load_and_save_data() -> None:
    train_test_info = s.loadmat(p.SUN_RGBD_DIRC.joinpath('SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat'))
    train_paths = format_split_paths(train_test_info['alltrain'][0])
    train_count, test_count = 0, 0
    for old_rgb_path, old_depth_path, bounding_box, seg_data in load_meta_info():
        old_rgb_path = relink_image_path(old_rgb_path)
        old_depth_path = relink_image_path(old_depth_path)

        if any([path in str(old_rgb_path) for path in train_paths]):
            save_dirc = p.SUN_RGBD_TRAIN_DIRC
            train_count += 1
            count = train_count
        else:
            save_dirc = p.SUN_RGBD_TEST_DIRC
            test_count += 1
            count = test_count

        save_dirc = save_dirc.joinpath(f'{count}')
        save_dirc.mkdir(exist_ok=True, parents=True)

        new_rgb_path = save_dirc.joinpath('rgb.png')
        new_depth_path = save_dirc.joinpath('depth.png')
        bounding_box_path = save_dirc.joinpath('bounding_box.npy')
        semantic_segs_path = save_dirc.joinpath('semantic_segs.npy')

        new_rgb_path.symlink_to(old_rgb_path)
        new_rgb_path.resolve()

        new_depth_path.symlink_to(old_depth_path)
        new_depth_path.resolve()

        np.save(bounding_box_path, bounding_box, allow_pickle=True)
        np.save(semantic_segs_path, seg_data, allow_pickle=True)


if __name__ == '__main__':
    load_and_save_data()
