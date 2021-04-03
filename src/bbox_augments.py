import numpy as np
from typing import Optional, List

"""Data augmentations for bounding boxes"""


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


def bbox_crop(bboxes: np.ndarray, min_x: int, min_y: int, max_x: int, max_y: int) -> np.ndarray:
    # TODO visibility threshold
    def crop(x1, y1, x2, y2) -> Optional[List[int]]:
        if x1 >= max_x or y1 >= max_y or x2 <= min_x or y2 <= min_y:
            return None
        new_x1 = max(0, x1 - min_x)
        new_y1 = max(0, y1 - min_y)
        new_x2 = min(max_x - min_x, x2 - min_x)
        new_y2 = min(max_y - min_y, y2 - min_y)
        return [new_x1, new_y1, new_x2, new_y2]

    new_bboxes = []
    for box in bboxes:
        if (new_box := crop(*box)) is not None:
            new_bboxes.append(new_box)
    return np.array(new_bboxes)
