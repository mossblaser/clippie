import pytest

from typing import Any

import torch

import numpy as np

import clip  # type: ignore

from pathlib import Path

from PIL import Image

from clippie.scripts.convert_weights_file import extract_weights

from clippie.image import centre_crop_and_resize

from clippie.model import (
    Weights,
    get_input_image_dimensions,
    encode_text,
    encode_image,
)

image_dir = Path(__file__).parent / "images"


@pytest.fixture(scope="module")
def clip_model_and_preprocess() -> tuple[Any, Any]:
    return clip.load(name="ViT-B/32", device="cpu")


@pytest.fixture(scope="module")
def clip_model(clip_model_and_preprocess: tuple[Any, Any]) -> Any:
    return clip_model_and_preprocess[0]


@pytest.fixture(scope="module")
def clip_preprocess(clip_model_and_preprocess: tuple[Any, Any]) -> Any:
    return clip_model_and_preprocess[1]


@pytest.fixture(scope="module")
def clippie_weights(clip_model: Any) -> Weights:
    return extract_weights(clip_model)


def test_encode_text_matches_clip(clip_model: Any, clippie_weights: Weights) -> None:
    text = "This is a test of CLIP and Clippie's equivalence"

    clippie_output = encode_text(text, clippie_weights.text_encoder)

    with torch.no_grad():
        clip_output = clip_model.encode_text(clip.tokenize(text)).numpy()

    # NB: float32 precision is quite limited
    assert np.allclose(clip_output, clippie_output, atol=1e-5)


def test_encode_image_matches_clip(
    clip_model: Any,
    clip_preprocess: Any,
    clippie_weights: Weights,
) -> None:
    im = Image.open(image_dir / "shocked_child.jpg")

    # Pre-resize to avoid differences in down-scaling filtering resulting in
    # misleadingly different outputs
    image_size = get_input_image_dimensions(clippie_weights.image_encoder)
    im = centre_crop_and_resize(im, image_size)

    clippie_output = encode_image(im, clippie_weights.image_encoder)

    with torch.no_grad():
        clip_output = clip_model.encode_image(clip_preprocess(im).unsqueeze(0)).numpy()

    # NB: float32 precision is quite limited
    assert np.allclose(clip_output, clippie_output, atol=1e-5)


def test_sanity_check_performance(
    clip_model: Any,
    clip_preprocess: Any,
    clippie_weights: Weights,
) -> None:
    # Three image/description pairs which ought to be easily distinguished
    dataset = {
        image_dir / "shocked_child.jpg": "A shocked child",
        image_dir / "soup.jpg": "Bowls of soup",
        image_dir / "messy_electronics.jpg": "Messy electronics",
    }
    images = [Image.open(f) for f in dataset.keys()]
    texts = list(dataset.values())

    image_encodings = encode_image(images, clippie_weights.image_encoder)
    text_encodings = encode_text(texts, clippie_weights.text_encoder)

    # Normalise the vectors
    image_encodings /= np.linalg.norm(image_encodings, axis=1, keepdims=True)
    text_encodings /= np.linalg.norm(text_encodings, axis=1, keepdims=True)

    # Cosine similarity
    similarity_matrix = image_encodings @ text_encodings.T

    print(similarity_matrix)

    # In each row, the diagonal (i.e. correct image/pair match) should be the
    # most highly ranked
    assert list(np.argmax(similarity_matrix, axis=1)) == list(range(len(dataset)))
