"""
Miscellaneous utiility functions.
"""

import numpy as np
from numpy.typing import NDArray


def split_axis(a: NDArray, axis: int, num_pieces: int):
    """
    Given an array, split the array into num_pieces equal sized pieces along
    that dimension, creating a new dimension 0 which indexes over the splits.

    As examples::

        >>> ar = np.arange(4 * 6).reshape(4, 6)
        >>> ar
        array([[ 0,  1,  2,  3,  4,  5],
              [ 6,  7,  8,  9, 10, 11],
              [12, 13, 14, 15, 16, 17],
              [18, 19, 20, 21, 22, 23]])

        >>> split_axis(ar, axis=0, num_pieces=2)
        array([[[ 0,  1,  2,  3,  4,  5],
                [ 6,  7,  8,  9, 10, 11]],
        <BLANKLINE>
              [[12, 13, 14, 15, 16, 17],
                [18, 19, 20, 21, 22, 23]]])
        >>> _.shape
        (2, 2, 6)

        >>> split_axis(ar, axis=1, num_pieces=2)
        array([[[ 0,  1,  2],
                [ 6,  7,  8],
                [12, 13, 14],
                [18, 19, 20]],
        <BLANKLINE>
              [[ 3,  4,  5],
                [ 9, 10, 11],
                [15, 16, 17],
                [21, 22, 23]]])
        >>> _.shape
        (2, 4, 3)

    .. note::

        This function will return a view if possible.
    """
    if axis < 0:
        axis += a.ndim

    if a.shape[axis] % num_pieces != 0:
        raise ValueError(
            f"Axis {axis} does not divide into {num_pieces} equal-sized pieces."
        )

    # Move the dimension to be split into the last dimension
    a = np.moveaxis(a, axis, -1)  #  -> (..., axis_to_split)

    # Split the last dimension up into the required number of pieces
    a = a.reshape(a.shape[:-1] + (num_pieces, a.shape[-1] // num_pieces))

    # The split above creates a new dimension indexing the splits at dimension
    # -2. Move that to position 0.
    a = np.moveaxis(a, -2, 0)

    # Restore the split axis to its original position (NB +1 since there's a new dimension now)
    a = np.moveaxis(a, -1, axis + 1)

    return a
