"""
Utility for loading and saving weights to disk.

The on-disk format is designed to be directly memory mapped meaning that when
the model is not in use the weights can be swapped out of memory and that only
weights which are actively in use will stay in RAM. Further, the on-disk format
does not require PyTorch to be installed, by contrast with the CLIP native
format.


Data format
===========

The file consists of two parts: the first part contains the raw array data and
the second a pickled representation of the `Weights` tuple with array objects
replaced with pointers to the raw data blocks.

All raw array data is 16kb-aligned in the native byte endianness of the system
which saved the data. The 16kb alignment means these arrays should start on
page boundaries on all common systems. I *think* that Numpy will do the correct
thing if loaded on a machine with the 'wrong' endianness, albeit at the cost of
performance. (But seriously, who has a big-endian machine to even test things
on?)

The pickled `Weights` object is followed by a 32-bit unsigned, little endian
integer indicating the number of bytes in the pickled data. All arrays in the
pickled structure are replaced with NDArrayOffset objects which include
everything needed to wrap this in an array.
"""


from typing import NamedTuple, IO, Any, cast

import pickle
import struct
import platform
import mmap

import numpy as np
from numpy.typing import NDArray, DTypeLike

from clippie.model import Weights


class NDArrayOffset(NamedTuple):
    """
    A stand-in for a numpy array stored at a particular location in the file.
    """

    offset: int
    shape: tuple[int, ...]
    dtype: np.dtype


def _dump_and_substitute_arrays(
    data: Any,
    file: IO[bytes],
    alignment: int,
) -> Any:
    """
    Given a NamedTuple-style container of numpy arrays and other generic Python
    values, return a new NamedTuple of the same type but with all NDArrays
    replaced with NDArrayOffset objects and the array data written to the
    provided file.
    """
    if isinstance(data, np.ndarray):
        # Pad to multiple of required alignment as necessary
        cur_offset = file.tell()
        offset = ((cur_offset + alignment - 1) // alignment) * alignment
        file.write(b"\0" * (offset - cur_offset))

        # Write the array
        file.write(data.tobytes())

        return NDArrayOffset(
            offset=offset,
            shape=data.shape,
            dtype=data.dtype,
        )
    elif isinstance(data, (tuple, list)):
        if hasattr(type(data), "_make"):
            make_t = type(data)._make  # type: ignore
        else:
            make_t = type(data)  # type: ignore
        return make_t(
            _dump_and_substitute_arrays(item, file, alignment) for item in data
        )
    else:
        # All other data types are serialised as-is
        return data


def _load_and_substitute_arrays(data: Any, memory_map: bytearray) -> Any:
    """
    Inverse of _dump_and_substitute_arrays. All loaded arrays will source their
    data directly from the passed in memory mapped region.
    """
    if isinstance(data, NDArrayOffset):
        ar = np.frombuffer(
            buffer=memory_map,
            offset=data.offset,
            count=np.prod(data.shape),
            dtype=data.dtype,
        )
        ar.flags.writeable = False
        ar = ar.reshape(data.shape)
        return ar
    elif isinstance(data, (tuple, list)):
        if hasattr(type(data), "_make"):
            make_t = type(data)._make  # type: ignore
        else:
            make_t = type(data)  # type: ignore
        return make_t(_load_and_substitute_arrays(item, memory_map) for item in data)
    else:
        # All other data types are serialised as-is
        return data


def dump(data: Weights, file: IO[bytes], alignment: int = 16 * 1024) -> None:
    """
    Dump the given Weights into the provided file with Numpy arrays being
    stored with the given alignment.
    """
    data = _dump_and_substitute_arrays(data, file, alignment)

    data_bytes = pickle.dumps(data)
    file.write(data_bytes)
    file.write(struct.pack("<I", len(data_bytes)))


def load(file: IO[bytes]) -> Weights:
    """
    Load a set of weights from the provided file with all arrays being memory
    mapped from the file.
    """
    extra_kwargs = {}
    if platform.system() != "Windows":
        extra_kwargs["prot"] = mmap.PROT_READ

    memory_map = mmap.mmap(file.fileno(), length=0, **extra_kwargs)

    data_len = struct.unpack("<I", memory_map[-4:])[0]
    data = pickle.loads(memory_map[-4 - data_len : -4])

    return cast(Weights, _load_and_substitute_arrays(data, cast(bytearray, memory_map)))
