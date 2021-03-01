import pathlib as pl
from typing import Tuple, Set, List

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


def strip_old_img_path(old_path: str) -> str:
    return '/'.join(old_path.split('/')[6:])


def relink_image_path(path: str) -> pl.Path:
    new_path = p.SUN_RGBD_DIRC.joinpath('SUNRGBD').joinpath(path)
    if not new_path.exists():
        raise ValueError(f'image path {new_path} doesn\'t exist')
    return new_path


def load_image_orderings_file(path) -> List[str]:
    with open(path, 'r') as f:
        files = list(f.readlines())
    return ['/'.join(x[:-1].split('/')[1:]) for x in files]


def load_meta_info() -> Tuple[str, str, np.ndarray]:
    mat_file = s.loadmat(p.SUN_RGBD_EXTRA_DIRC.joinpath('SUNRGBDMeta2DBB_v2.mat'))
    paths = mat_file['SUNRGBDMeta2DBB']
    rgb_paths, depth_paths, bounding_boxes = paths['rgbpath'][0], paths['depthpath'][0], paths['groundtruth2DBB'][0]

    if len(rgb_paths) != len(depth_paths) != len(bounding_boxes):
        raise ValueError('number of RGB image paths, depth data paths, and bounding boxes must be the same')

    groups = [rgb_paths, depth_paths, bounding_boxes]
    for rgb_path, depth_path, bounding_box in zip(*groups):
        yield rgb_path[0], depth_path[0], bounding_box


def format_split_paths(paths: np.ndarray) -> Set[str]:
    return set(('/'.join(path[0].split('/')[5:]) for path in paths))


def load_and_save_data() -> None:
    train_img_ordering = load_image_orderings_file(p.SUN_RGBD_EXTRA_DIRC.joinpath('sunrgbd_training_images.txt'))
    test_img_ordering = load_image_orderings_file(p.SUN_RGBD_EXTRA_DIRC.joinpath('sunrgbd_testing_images.txt'))
    semantic_img_dirc = p.SUN_RGBD_EXTRA_DIRC.joinpath('sunrgbd_train_test_labels')
    train_test_info = s.loadmat(p.SUN_RGBD_DIRC.joinpath('SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat'))
    train_paths = format_split_paths(train_test_info['alltrain'][0])
    train_count, test_count = 0, 0
    for old_rgb_path, old_depth_path, bounding_box in load_meta_info():
        # TODO get semantic image path (if train adjust index), save here
        # TODO redo if image in train paths
        old_rgb_path = strip_old_img_path(old_rgb_path)
        old_depth_path = strip_old_img_path(old_depth_path)

        if old_rgb_path in train_img_ordering:  # any([path in str(old_rgb_path) for path in train_paths]):
            save_dirc = p.SUN_RGBD_TRAIN_DIRC
            train_count += 1
            count = train_count

            # 5050 testing images before training images
            semantic_segs_id = train_img_ordering.index(old_rgb_path) + 5050
        else:
            save_dirc = p.SUN_RGBD_TEST_DIRC
            test_count += 1
            count = test_count
            semantic_segs_id = test_img_ordering.index(old_rgb_path)

        semantic_segs_id += 1  # 0 to 1 based index
        sem_segs_path = semantic_img_dirc.joinpath(f'img-{semantic_segs_id:06d}.png')

        old_rgb_path = relink_image_path(old_rgb_path)
        old_depth_path = relink_image_path(old_depth_path)

        save_dirc = save_dirc.joinpath(f'{count}')
        save_dirc.mkdir(exist_ok=True, parents=True)

        new_rgb_path = save_dirc.joinpath('rgb.png')
        new_depth_path = save_dirc.joinpath('depth.png')
        new_sem_seg_path = save_dirc.joinpath('semantic_segs.png')
        bounding_box_path = save_dirc.joinpath('bounding_box.npy')

        new_rgb_path.symlink_to(old_rgb_path)
        new_rgb_path.resolve()

        new_depth_path.symlink_to(old_depth_path)
        new_depth_path.resolve()

        new_sem_seg_path.symlink_to(sem_segs_path)
        new_sem_seg_path.resolve()

        np.save(bounding_box_path, bounding_box, allow_pickle=True)


if __name__ == '__main__':
    load_and_save_data()
