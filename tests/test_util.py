import pytest

import numpy as np
from numpy.typing import NDArray

from clippie.util import split_axis


@pytest.mark.parametrize(
    "axis, num_pieces, ar, exp",
    [
        # 1D input
        (
            0,
            3,
            np.arange(9),
            np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
        ),
        # 2D input
        (
            0,
            2,
            np.arange(12).reshape(4, 3),
            np.array(
                [
                    [
                        [0, 1, 2],
                        [3, 4, 5],
                    ],
                    [
                        [6, 7, 8],
                        [9, 10, 11],
                    ],
                ]
            ),
        ),
        (
            1,
            2,
            np.arange(12).reshape(3, 4),
            np.array(
                [
                    [
                        [0, 1],
                        [4, 5],
                        [8, 9],
                    ],
                    [
                        [2, 3],
                        [6, 7],
                        [10, 11],
                    ],
                ]
            ),
        ),
        # Negative axis
        (
            -1,
            2,
            np.arange(12).reshape(3, 4),
            np.array(
                [
                    [
                        [0, 1],
                        [4, 5],
                        [8, 9],
                    ],
                    [
                        [2, 3],
                        [6, 7],
                        [10, 11],
                    ],
                ]
            ),
        ),
    ],
)
def test_split_axis(axis: int, num_pieces: int, ar: NDArray, exp: NDArray) -> None:
    actual = split_axis(ar, axis=axis, num_pieces=num_pieces)

    assert np.array_equal(actual, exp)
    assert np.shares_memory(ar, actual)


def test_split_axis_invalid() -> None:
    with pytest.raises(ValueError):
        split_axis(np.arange(9), axis=0, num_pieces=2)
