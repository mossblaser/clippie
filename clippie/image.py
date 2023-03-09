"""
Utilities for loading images for use with the CLIP model.

This loosely corresponds to the image 'preprocessing' function provided in the
reference CLIP implementation.
"""

from PIL import Image

import numpy as np
from numpy.typing import NDArray


CLIP_RGB_NORMALISATION_MEAN = np.array((0.48145466, 0.4578275, 0.40821073))
CLIP_RGB_NORMALISATION_STD = np.array((0.26862954, 0.26130258, 0.27577711))
"""
The mean and standard deviation to which images have their R, G and B values
scaled to prior to encoding. Values taken from the CLIP codebase.
"""


def centre_crop_and_resize(im: Image.Image, size: int) -> Image.Image:
    """
    Return the center-cropped image, scaled down the size x size pixels.
    """
    d = min(im.size)
    x = (im.size[0] - d) / 2
    y = (im.size[1] - d) / 2

    # Crop to central square and resize
    im = im.resize(
        size=(size, size),
        resample=Image.Resampling.BICUBIC,
        box=(x, y, x + d, y + d),
    )

    return im


def image_to_scaled_array(im: Image.Image) -> NDArray:
    """
    Take a PIL image and convert it into (height, width, 3) array with values
    shifted and scaled as done in the reference CLIP implementation.
    """
    ima = np.array(im.convert("RGB")).astype(np.float32) / 255.0

    # Apply the scaling factors used by the reference CLIP implementation
    #
    # This (I assume) normalises the "average" image to zero mean and unit
    # standard deviation.
    ima -= CLIP_RGB_NORMALISATION_MEAN
    ima /= CLIP_RGB_NORMALISATION_STD

    return ima
