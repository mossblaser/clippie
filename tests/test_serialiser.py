import pytest

from typing import Any, Iterable

from pathlib import Path

from itertools import zip_longest

from numpy.typing import NDArray
import numpy as np

from clippie.model import (
    Weights,
    TextEncoderWeights,
    ImageEncoderWeights,
    ResidualAttentionBlockWeights,
    LayerNormalisationWeights,
)

from clippie.serialiser import load, dump


@pytest.fixture
def fake_weights() -> Weights:
    value = 0

    def make_fake() -> NDArray:
        nonlocal value
        value += 1
        return np.full((2, 3, 4), value)

    # Dummy values, not even valid dimensions(!)
    return Weights(
        text_encoder=TextEncoderWeights(
            token_embedding_lut=make_fake(),
            positional_encoding=make_fake(),
            transformer=[
                ResidualAttentionBlockWeights(
                    pre_attention_layer_norm=LayerNormalisationWeights(
                        weights=make_fake(),
                        biases=make_fake(),
                        eps=1e-8,
                    ),
                    n_heads=8,
                    qkv_proj_weights=make_fake(),
                    qkv_proj_biases=make_fake(),
                    multi_head_output_proj_weights=make_fake(),
                    multi_head_output_proj_biases=make_fake(),
                    attention_mask=True,
                    pre_mpl_layer_norm=LayerNormalisationWeights(
                        weights=make_fake(),
                        biases=make_fake(),
                        eps=1e-8,
                    ),
                    mlp_input_weights=make_fake(),
                    mlp_input_biases=make_fake(),
                    mlp_output_weights=make_fake(),
                    mlp_output_biases=make_fake(),
                )
                for _ in range(3)
            ],
            transformer_output_norm=LayerNormalisationWeights(
                weights=make_fake(),
                biases=make_fake(),
                eps=1e-8,
            ),
            output_projection_weights=make_fake(),
        ),
        image_encoder=ImageEncoderWeights(
            convolution_weights=make_fake(),
            class_value=make_fake(),
            positional_encoding=make_fake(),
            pre_transformer_layer_norm=LayerNormalisationWeights(
                weights=make_fake(),
                biases=make_fake(),
                eps=1e-8,
            ),
            transformer=[
                ResidualAttentionBlockWeights(
                    pre_attention_layer_norm=LayerNormalisationWeights(
                        weights=make_fake(),
                        biases=make_fake(),
                        eps=1e-8,
                    ),
                    n_heads=12,
                    qkv_proj_weights=make_fake(),
                    qkv_proj_biases=make_fake(),
                    multi_head_output_proj_weights=make_fake(),
                    multi_head_output_proj_biases=make_fake(),
                    attention_mask=False,
                    pre_mpl_layer_norm=LayerNormalisationWeights(
                        weights=make_fake(),
                        biases=make_fake(),
                        eps=1e-8,
                    ),
                    mlp_input_weights=make_fake(),
                    mlp_input_biases=make_fake(),
                    mlp_output_weights=make_fake(),
                    mlp_output_biases=make_fake(),
                )
                for _ in range(4)
            ],
            post_transformer_layer_norm=LayerNormalisationWeights(
                weights=make_fake(),
                biases=make_fake(),
                eps=1e-8,
            ),
            output_projection_weights=make_fake(),
        ),
    )


def iter_weights(weights: Weights) -> Iterable[tuple[str, Any]]:
    """Iterate over (label, value) pairs in a weights dictionary."""
    to_yield = [("weights", weights)]
    while to_yield:
        prefix, weights = to_yield.pop()
        if isinstance(weights, list):
            to_yield.extend((f"{prefix}[{i}]", x) for i, x in enumerate(weights))
        elif hasattr(weights, "_fields"):
            to_yield.extend(
                (f"{prefix}.{field}", getattr(weights, field))
                for field in weights._fields
            )
        else:
            yield (prefix, weights)


def test_roundtrip(fake_weights: Weights, tmp_path: Path) -> None:
    filename = tmp_path / "weights"

    with filename.open("wb") as f:
        dump(fake_weights, f)

    with filename.open("rb") as f:
        loaded_weights = load(f)

    for (a_label, a_value), (b_label, b_value) in zip_longest(
        iter_weights(fake_weights), iter_weights(loaded_weights), fillvalue=("", None)
    ):
        assert a_label == b_label
        print(a_label)

        assert type(a_value) == type(b_value)
        if isinstance(a_value, np.ndarray) and isinstance(b_value, np.ndarray):
            assert a_value.dtype == b_value.dtype
            assert np.array_equal(a_value, b_value)
        else:
            assert a_value == b_value
