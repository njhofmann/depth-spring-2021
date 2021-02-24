import torchvision.transforms as tvt

SUN_RGBD_TRANSFORMS = tvt.Compose([
    tvt.C
])


class AwareColorJittering:

    """Color jittering for RGBD images, applies jittering only to the RGB part of the image"""

    def __init__(self, brightness, contrast, saturation, hue):
        self.jitter = tvt.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, sample):
        # TODO check img shape, sample input
        img = sample[0]
        if len(img.shape[-1]) == 4:
            img[:3] = self.jitter(img[:3])
        elif len(img.shape[-1] == 3):
            img = self.jitter(img)

        return img, sample[1]