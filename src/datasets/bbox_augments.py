import numpy as np
from typing import Optional, Tuple

"""Data augmentations for bounding boxes"""


def adjust_torch_crop_params(crop_params: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    return (crop_params[1],
            crop_params[0],
            crop_params[1] + crop_params[3],
            crop_params[0] + crop_params[2])


def bbox_hflip(bboxes: np.ndarray, cols: int) -> np.ndarray:
    new_x1 = cols - bboxes[:, 2]
    new_x2 = cols - bboxes[:, 0]
    bboxes[:, 0] = new_x1
    bboxes[:, 2] = new_x2
    return bboxes


def bbox_vflip(bboxes: np.ndarray, rows: int) -> np.ndarray:
    new_y1 = rows - bboxes[:, 3]
    new_y2 = rows - bboxes[:, 1]
    bboxes[:, 1] = new_y1
    bboxes[:, 3] = new_y2
    return bboxes


def box_area(box: Tuple[int, int, int, int]) -> int:
    return (box[3] - box[1]) * (box[2] - box[0])


def is_box_visible(old_box: Tuple[int, int, int, int], new_box: Tuple[int, int, int, int], visibility: float) -> bool:
    return box_area(old_box) > (box_area(new_box) * visibility)


def bbox_crop(bboxes: np.ndarray, min_x: int, min_y: int, max_x: int, max_y: int, visibility: float = .5) -> np.ndarray:
    def crop(x1, y1, x2, y2) -> Optional[Tuple[int, int, int, int]]:
        if x1 >= max_x or y1 >= max_y or x2 <= min_x or y2 <= min_y:
            return None
        new_x1 = max(0, x1 - min_x)
        new_y1 = max(0, y1 - min_y)
        new_x2 = min(max_x - min_x, x2 - min_x)
        new_y2 = min(max_y - min_y, y2 - min_y)
        return new_x1, new_y1, new_x2, new_y2

    new_bboxes = []
    for box in bboxes:
        if (new_box := crop(*box)) is not None and is_box_visible(box, new_box, visibility):
            new_bboxes.append(new_box)
    return np.array(new_bboxes)
