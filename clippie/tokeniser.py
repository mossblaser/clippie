"""
An implementation of the Byte-Pair Tokenisation scheme used by CLIP.

The basic scheme was Originally described in 'Neural Machine Translation of
Rare Words with Subword Units', 2016, Sennrich, Haddow and Birch.

Numerous implementation details necessarily mimic (but are not copied from) the
OpenAI CLIP source to ensure compatibility.
"""

from typing import Iterable

from pathlib import Path

import gzip

import pickle

import regex

import importlib

# Read the vocab and byte-pair encoding merge lists.
#
# This vocab file is derrived from the `clip/bpe_simple_vocab_16e6.txt.gz` file
# included in the CLIP repository (see
# :py:mod:`clippie.scripts.convert_vocab_file` for the conversion process and
# rationale).
#
# VOCAB is a list containing the byte string encoded by each possible token value.
#
# BPE_MERGE_LIST is a list of byte pairs in the order in which they should
# be merged during byte pair encoding.
with importlib.resources.open_binary("clippie", "data", "vocab.pickle.gz") as f:
    VOCAB, BPE_MERGE_LIST = pickle.load(gzip.GzipFile(fileobj=f))

# Mappings between tokens and their corresponding values
VALUE_TO_TOKEN: dict[bytes, int] = {value: token for token, value in enumerate(VOCAB)}
TOKEN_TO_VALUE: dict[int, bytes] = {token: value for token, value in enumerate(VOCAB)}

# The sentinel value used to indicate the end of a word in the vocab file
#
# NB: This can never appear 'accidentally' in the vocab since CLIP never
# performs byte-pair encoding across special characters.
WORD_SENTINEL = b"</w>"

# Sentinel values in the vocab indicating the start and end of text.
#
# NB: These strings can't appear 'accidentally' in the vocab since CLIP never
# performs byte-pair encoding across special characters.
START_OF_TEXT_SENTINEL = b"<|startoftext|>"
END_OF_TEXT_SENTINEL = b"<|endoftext|>"

# The token value encoding the above sentinel values.
START_OF_TEXT_TOKEN = VALUE_TO_TOKEN[START_OF_TEXT_SENTINEL]
END_OF_TEXT_TOKEN = VALUE_TO_TOKEN[END_OF_TEXT_SENTINEL]

# For each byte-pair merge available gives the rank (with the highest priority
# merges having the lowest number).
BPE_MERGE_RANK: dict[tuple[bytes, bytes], int] = {
    pair: rank for rank, pair in enumerate(BPE_MERGE_LIST)
}


def byte_pair_encode_word(string: bytes) -> list[bytes]:
    """
    Given a string containing a single 'word' (from the perspective of CLIP's
    tokenizer), perform byte-pair encoding, returning the list of merged bytes.
    """
    # Split the string into individual bytes, with the final byte appended with an
    # end-of-word marker.
    parts = [bytes((c,)) for c in string[:-1]] + [string[-1:] + WORD_SENTINEL]

    while len(parts) > 1:
        # Find the best possible merge given the current state of the string
        best_merge = min(
            zip(parts, parts[1:]),
            key=lambda pair: BPE_MERGE_RANK.get(pair, len(BPE_MERGE_RANK)),
        )

        # If no merges are available, we're done
        if best_merge not in BPE_MERGE_RANK:
            break

        # Substitute that merge wherever it occurs
        new_word_parts = []
        while parts:
            if tuple(parts[:2]) == best_merge:
                new_word_parts.append(parts.pop(0) + parts.pop(0))
            else:
                new_word_parts.append(parts.pop(0))
        parts = new_word_parts

    return parts


# The following regex matches blocks of input which may be byte-pair encoded as
# a group (e.g. words).
#
# See also :py:data:`clip.simple_tokenizer.SimpleTokenizer.pat`.
BPE_BLOCK_RE = regex.compile(
    "|".join(
        [
            # Special case for (English) apostrophe usages. Always tokenized
            # separately from the rest of the word, but along with the
            # apostrophe. (In other usages, punctuation is always split into a
            # separate token).
            r"'s",
            r"'t",
            r"'re",
            r"'ve",
            r"'m",
            r"'ll",
            r"'d",
            # Sequence of Unicode letter-class characters
            r"[\p{L}]+",
            # Individual Unicode number-class characters (i.e. digits are
            # always Tokenized individually).
            r"[\p{N}]",
            # Other sequences non-space/letter/number characters
            r"[^\s\p{L}\p{N}]+",
        ]
    ),
    regex.IGNORECASE,
)


def encode(string: str) -> Iterable[int]:
    """
    Tokenise a given string using the CLIP tokeniser.

    Generates a series of token values.

    .. note::
        In the CLIP implementation, additional text encoding issues (e.g. bad
        text encodings or embedded HTML escape sequences) are automatically
        stripped out. Here, however, we assume better behaved inputs and so all
        we skip all that.

    .. note::
        Unlike the CLIP tokenizer, we do not have special-case treatment of the
        START_OF_TEXT_SENTINEL and END_OF_TEXT_SENTINEL. Including these in the
        string as iterals will result in them being tokenised as ordinary text
        instead. Add START_OF_TEXT_TOKEN and END_OF_TEXT_TOKEN to your token
        stream manually instead if needed.
    """

    # CLIP's tokenisation scheme is case-insenstive
    string = string.lower()

    # NB: Whitespace normalisation is also skipped since the regex skips all
    # whitespace implicitly.
    for word in BPE_BLOCK_RE.findall(string):
        # NB: The CLIP tokenizer acts on UTF-8 encoded strings rather than
        # operating on unicode codepoints directly to keep the size of the
        # vocabluary under control.
        for part in byte_pair_encode_word(word.encode("utf-8")):
            yield VALUE_TO_TOKEN[part]


def decode(tokens: Iterable[int]) -> bytes:
    """
    Decode a sequence of tokens into a (byte) string.

    This function is provided for debugging purposes only. Tokenisation is
    lossy:

    * Whitespace will be changed and possibly added inside 'words'
    * Because of the above -- and the fact that tokenisation acts at the byte
      level -- the generated values may not be valid UTF-8 anymore so a bytes
      object is returned.
    """
    return b"".join(TOKEN_TO_VALUE[t] for t in tokens).replace(WORD_SENTINEL, b" ")
