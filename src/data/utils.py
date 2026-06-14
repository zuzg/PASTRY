import math
from enum import Enum, auto

import numpy as np
from skimage.filters import gaussian
from skimage.transform import rescale
from torch import Tensor, nn


class ImageMode(Enum):
    GRAYSCALE = auto()
    CHANNELS_FIRST = auto()
    CHANNELS_LAST = auto()


def blur(img: np.ndarray, mode: ImageMode, *, sigma=1.0) -> np.ndarray:
    to_blur = img.copy()
    if mode == ImageMode.GRAYSCALE:
        return gaussian(to_blur, sigma=sigma)

    if mode == ImageMode.CHANNELS_LAST:
        to_blur = np.moveaxis(to_blur, 2, 0)
    for i, channel in enumerate(to_blur):
        to_blur[i] = gaussian(channel, sigma=sigma)
    return np.moveaxis(to_blur, 0, 2) if mode == ImageMode.CHANNELS_LAST else to_blur


def downscale(img: np.ndarray, ratio: int, mode: ImageMode) -> np.ndarray:
    if mode == ImageMode.GRAYSCALE:
        return rescale(img, scale=1.0 / ratio)

    downscaled_channels = []
    if mode == ImageMode.CHANNELS_LAST:
        img = np.moveaxis(img, 2, 0)
    for channel in img:
        downscaled_channels.append(rescale(channel, scale=1.0 / ratio))
    out_ch_ax = -1 if mode == ImageMode.CHANNELS_LAST else 0
    return np.stack(downscaled_channels, axis=out_ch_ax)


def downsample(img: np.ndarray, ratio: int, mode: ImageMode) -> np.ndarray:
    """Take every nth (n = `ratio`) pixel in spatial dimensions."""
    if len(img.shape) not in (2, 3):
        raise ValueError(
            f"Invalid img input shape, should be 2 or 3, but got {img.shape}"
        )
    if mode == ImageMode.CHANNELS_FIRST:
        return img[..., ratio // 2 :: ratio, ratio // 2 :: ratio]
    else:
        return img[ratio // 2 :: ratio, ratio // 2 :: ratio, ...]


def apply_walds_protocol(img: np.ndarray, ratio: int, mode: ImageMode) -> np.ndarray:
    # Window blur size is inferred from sigma in skimage (in orig paper it is 8 by 8)
    sigma = math.sqrt(1 / (2 * 2.7725887 / ratio**2))
    blurred = blur(img, mode, sigma=sigma)
    downsampled = downsample(blurred, ratio, mode)
    return downsampled


def create_synthetic_lr(hr_image: Tensor, factor: int) -> Tensor:
    hr_image_bchw = hr_image.permute(2, 0, 1).unsqueeze(0)
    lr_image_bchw = nn.functional.avg_pool2d(
        hr_image_bchw, kernel_size=factor, stride=factor
    )
    lr_image_chw = lr_image_bchw.squeeze(0)
    lr_image_hwB = lr_image_chw.permute(1, 2, 0)
    return lr_image_hwB
