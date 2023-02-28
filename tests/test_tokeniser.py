import pytest

from clip.simple_tokenizer import SimpleTokenizer  # type: ignore

from clippie.scripts.convert_vocab_file import _codepoint_binary_decode
from clippie.tokeniser import (
    encode,
    decode,
    VALUE_TO_TOKEN,
    BPE_MERGE_RANK,
    WORD_SENTINEL,
    START_OF_TEXT_SENTINEL,
    END_OF_TEXT_SENTINEL,
)


@pytest.fixture
def clip_tokeniser() -> SimpleTokenizer:
    return SimpleTokenizer()


def test_bpe_merge_list_identical(clip_tokeniser: SimpleTokenizer) -> None:
    # Verify that the byte-pair encoding merge lists exactly match CLIP
    assert BPE_MERGE_RANK == {
        (_codepoint_binary_decode(a), _codepoint_binary_decode(b)): rank
        for (a, b), rank in clip_tokeniser.bpe_ranks.items()
    }


def test_vocab_identical(clip_tokeniser: SimpleTokenizer) -> None:
    # Verify that the vocab exactly matches CLIP
    assert VALUE_TO_TOKEN == {
        _codepoint_binary_decode(value): token
        for value, token in clip_tokeniser.encoder.items()
    }


@pytest.mark.parametrize(
    "string",
    [
        # Empty
        "",
        " ",
        # Single word (possibly multiple tokens)
        "I",
        "Hello",
        "Heathcote",
        # Multiple words
        "Hello world",
        # Numbers
        "A123B",
        # Punctuation
        "A+B!",
        # The word sentinel should be ignored
        WORD_SENTINEL.decode("utf-8"),
    ],
)
def test_tokenisation(string: str, clip_tokeniser: SimpleTokenizer) -> None:
    assert list(encode(string)) == clip_tokeniser.encode(string)


@pytest.mark.parametrize("string", [START_OF_TEXT_SENTINEL, END_OF_TEXT_SENTINEL])
def test_tokenisation_special_cases(
    string: bytes, clip_tokeniser: SimpleTokenizer
) -> None:
    # Make sure we *don't* produce the sentinel tokens when presented with them
    # as input strings
    assert len(list(encode(string.decode("utf-8")))) > 1
