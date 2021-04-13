from typing import List, Tuple
import torch as t
import numbers
import torchvision.transforms.functional as f


"""Modified version of torchvision's center crop method to get bounding box coordinates of the resulting center crop 
instead of a cropped image - from: https://github.com/pytorch/vision/blob/master/torchvision/transforms/functional.py"""


def center_crop(img: t.Tensor, output_size: List[int]) -> Tuple[int, int, int, int]:
    """Crops the given image at the center.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    If image size is smaller than output size along any edge, image is padded with 0 and then center cropped.
    Args:
        img (PIL Image or Tensor): Image to be cropped.
        output_size (sequence or int): (height, width) of the crop box. If int or sequence with single int,
            it is used for both directions.
    Returns:
        PIL Image or Tensor: Cropped image.
    """
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    elif isinstance(output_size, (tuple, list)) and len(output_size) == 1:
        output_size = (output_size[0], output_size[0])

    image_width, image_height = f._get_image_size(img)
    crop_height, crop_width = output_size

    if crop_width > image_width or crop_height > image_height:
        padding_ltrb = [
            (crop_width - image_width) // 2 if crop_width > image_width else 0,
            (crop_height - image_height) // 2 if crop_height > image_height else 0,
            (crop_width - image_width + 1) // 2 if crop_width > image_width else 0,
            (crop_height - image_height + 1) // 2 if crop_height > image_height else 0,
        ]
        img = f.pad(img, padding_ltrb, fill=0)  # PIL uses fill value 0
        image_width, image_height = f._get_image_size(img)
        if crop_width == image_width and crop_height == image_height:
            return 0, 0, image_height, image_width

    crop_top = int(round((image_height - crop_height) / 2.))
    crop_left = int(round((image_width - crop_width) / 2.))
    crop_bottom = crop_top + crop_height
    crop_right = crop_left + crop_width
    return crop_top, crop_left, crop_bottom, crop_right
