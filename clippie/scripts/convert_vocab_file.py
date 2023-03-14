"""
This script converts the vocab file used by CLIP into a somewhat more sensible
format which is used by clippie.

The tokeniser used by CLIP uses a somewhat quirky vocabulary ordering (defined
implicitly by :py:mod:`clip.simple_tokenizer`). This script recreates this
ordering and writes it to a file for use by clippie.

The format used by clippie is a Python pickle file containing a value of type:

    tuple[
        list[bytes],                # Vocabulary
        list[tuple[bytes, bytes]],  # Byte pair encoding merges
    ]

The first list contains the vocabulary, the second contains the byte pair
encoding merges in order of descending order of application priority.
"""

from typing import Iterable, IO

from collections import OrderedDict

from itertools import islice

from argparse import ArgumentParser

from pathlib import Path

import gzip

import pickle

from clippie.tokeniser import (
    WORD_SENTINEL,
    START_OF_TEXT_SENTINEL,
    END_OF_TEXT_SENTINEL,
)


def _make_byte_to_codepoint_mapping() -> OrderedDict[int, str]:
    """
    This function returns a mapping from byte values to corresponding Unicode
    character as used by CLIP's vocab file.

    The CLIP vocab file uses a somewhat quirky scheme to encode arbitrary
    binary data using a relatively arbitrary set of printable Unicode
    characters. This mapping has the useful properties that as well as being
    printable, the basic printable ASCII characters are encoded as themselves.

    The returned dictionary is deliberately ordered to match the implementation
    in CLIP (see :py:func:`clip.simple_tokenizer.bytes_to_unicode`) since this
    ordering is used to map byte values to token numbers in the CLIP tokenizer.
    """
    mapping: OrderedDict[int, str] = OrderedDict()

    # Printable ASCII characters are represented as themselves (NB: does not
    # include 32=space! as space is considered non-printable in this context)
    for i in range(33, 126 + 1):
        mapping[i] = chr(i)

    # Where printable, Unicode codepoints in the range 128-255 are used to
    # represent those byte values.
    #
    # 128-159 (inclusive) are in the 'C1 control codes' block and are not
    # printable.
    #
    # 160 is a non-breaking space and thus is considered non-printable.
    #
    # From 161 to 255 are printable characters (from the Latin-1 supliment)
    # with the exception of code point 173 ('soft hyphen' -- a hyphenation
    # hint).
    for i in range(161, 172 + 1):
        mapping[i] = chr(i)
    for i in range(174, 255 + 1):
        mapping[i] = chr(i)

    # So far, then, we've managed to encode 189 of the possible byte values
    # using the unicode symbol with the unicode code point of the same number.
    # All that remains is to pick code points for the remaining byte values. We
    # do this by assigning them sequentially from the Latin Extended-A code
    # block which runs from code point 256 to 383 (inclusive) an is made up of
    # only printable characters:
    next_free_codepoint = 256
    for i in range(256):
        if i not in mapping:
            mapping[i] = chr(next_free_codepoint)
            next_free_codepoint += 1

    return mapping


# Forward and backward mappings between byte values and unicode codepoints used
# to represent those bytes in the CLIP vocab file.
_BYTE_TO_CODEPOINT = _make_byte_to_codepoint_mapping()
_CODEPOINT_TO_BYTE = {cp: b for b, cp in _BYTE_TO_CODEPOINT.items()}


def _codepoint_binary_decode(encoded: str) -> bytes:
    """
    Decode the binary data encoded by a given string of unicode codepoints in a
    CLIP vocab file.
    """
    return bytes(_CODEPOINT_TO_BYTE[codepoint] for codepoint in encoded)


# For each byte value (0-255 inclusive), the token value which is used to
# represent that byte literal.
_BYTE_VALUE_TO_TOKEN = list(_BYTE_TO_CODEPOINT.keys())


def _read_clip_byte_pair_merges(f: IO[bytes]) -> Iterable[tuple[bytes, bytes]]:
    """
    Lazily load the CLIP byte-pair encodings from an open (non-gzipped) file.

    Generates a series of (first_word, second_word) pairs in the order they're
    contained in the file.
    """
    # NB: First line is a comment which must be skipped
    f.readline()
    while True:
        line = f.readline().decode("utf-8").rstrip("\n")
        if line == "":
            break

        # Each line in the vocab file contains a 'byte-pair' (i.e. pair of
        # words which can be merged), with pairs of decreasing frequency coming
        # later in the file.
        first_part_encoded, _, second_part_encoded = line.partition(" ")

        first_part = _codepoint_binary_decode(first_part_encoded)
        second_part = _codepoint_binary_decode(second_part_encoded)

        yield (first_part, second_part)


def _generate_vocab(f: IO[bytes]) -> tuple[list[bytes], list[tuple[bytes, bytes]]]:
    """
    Generate the vocab using byte-pairs read from a CLIP byte pair encodings
    file.

    Returns the vocab as a list and the corresponding byte-pair encoding merge
    list.
    """
    # The first 256 codes correspond to a byte of equivalent value.
    #
    # These 256 values are not, however, assigned in the obvious order (where
    # a byte of 0 corresponds to token 0 and a byte of 1 has token 1...).
    # Instead they're ordered according to the order in which
    # clip.simple_tokenizer.bytes_to_unicode happened to return its encodings.
    vocab = [bytes((n,)) for n in _BYTE_VALUE_TO_TOKEN]

    # The next 256 codes correspond to a byte of equivalent value followed by
    # WORD_SENTINEL meaning 'end of word', in the same funny order.
    vocab += [bytes((n,)) + WORD_SENTINEL for n in _BYTE_VALUE_TO_TOKEN]

    # The next 48894 codes represent the first 48894 byte pair encodings in the
    # provided vocab file. NB: This file's first line is a comment and so is
    # skipped.
    #
    # NB: The number'48894' appears to be derived from the statement in the
    # CLIP paper that:
    #
    #     The transformer operates on a lower-cased byte pair encoding (BPE)
    #     representation of the text with a 49,152 vocab size.
    #
    # However, the published CLIP source code apparently fails to account for
    # the 256 extra "byte+WORD_SENTINEL" words added in the step above. As a result,
    # contrary to the statement in the paper, the vocab actually contains 49408
    # words.
    merges = []
    for first, second in islice(_read_clip_byte_pair_merges(f), 48894):
        merges.append((first, second))
        vocab.append(first + second)

    # The last two words in the vocab correspond to the special
    # sentinels for the beginning and end of text.
    vocab.append(START_OF_TEXT_SENTINEL)
    vocab.append(END_OF_TEXT_SENTINEL)

    return vocab, merges


def main():
    parser = ArgumentParser(
        "Convert a CLIP vocab file into the format used by clippie."
    )
    parser.add_argument("clip_bpe_file", type=Path, help="CLIP vocab file to convert")
    parser.add_argument("output", type=Path, help="Clippie vocab file to generate")
    args = parser.parse_args()

    with gzip.open(args.clip_bpe_file, "rb") as clip_bpe_file:
        vocab, merges = _generate_vocab(clip_bpe_file)

    with gzip.open(args.output, "wb") as output:
        pickle.dump((vocab, merges), output)


if __name__ == "__main__":
    main()
