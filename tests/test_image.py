import pytest

from pathlib import Path

from PIL import Image

import numpy as np

from clippie.image import centre_crop_and_resize


TEST_IMAGE_DIR = Path(__file__).parent / "images"


@pytest.mark.parametrize(
    "filename",
    [
        TEST_IMAGE_DIR / "centre_crop_test_landscape.png",
        TEST_IMAGE_DIR / "centre_crop_test_portrait.png",
    ],
)
def test_centre_crop_and_resize(filename: Path) -> None:
    with Image.open(filename) as im:
        im = centre_crop_and_resize(im.convert("RGB"), 18)
        ima = np.array(im)

    # Check top left corner is red (NB ignore pixels at edges which may be
    # slightly off due to filtering
    assert np.all(ima[1 : 9 - 1, 1 : 9 - 1] == np.array((255, 0, 0)))

    # Check rest of image is white (NB excluding edge most pixels again due to
    # filtering)
    assert np.all(ima[9 + 1 : -1, 1:-1] == np.array((255, 255, 255)))
    assert np.all(ima[1:-1, 9 + 1 : -1] == np.array((255, 255, 255)))
