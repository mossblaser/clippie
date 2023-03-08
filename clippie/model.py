"""
A numpy based implementation of the OpenAI CLIP model's forward pass.
"""

from typing import NamedTuple

from numpy.typing import NDArray

import numpy as np

from clippie import tokeniser

from clippie.nn import (
    embedding,
    layer_normalisation,
    make_attention_mask,
    multi_head_attention,
    approximate_gelu,
)


class LayerNormalisationWeights(NamedTuple):
    """Weights for a ptyhonlayer_normalisation."""
    weights: NDArray
    biases: NDArray
    eps: float


class ResidualAttentionBlockWeights(NamedTuple):
    """
    The weights for a single residual attention block within a transformer.
    """

    pre_attention_layer_norm: LayerNormalisationWeights  # (dim)
    """
    Layer normlisation for inputs to the attention block.
    """

    n_heads: int
    """Number of attention heads."""

    qkv_proj_weights: NDArray  # (dim, dim*3)
    qkv_proj_biases: NDArray  # (dim*3)
    """
    Weights and biases for the Q, K and V projections, concatenated along 0th
    dimension in that order.
    """

    multi_head_output_proj_weights: NDArray  # (dim, dim)
    multi_head_output_proj_biases: NDArray  # (dim)
    """
    Weights and biases for the projection from the concatenated attention head
    outputs to the combined final output.
    """

    attention_mask: bool
    """Should an causal attention mask be applied?"""

    pre_mpl_layer_norm: LayerNormalisationWeights  # (dim)
    """
    Layer normalisation values for the inputs to the multi-layer perceptron.
    """

    mlp_input_weights: NDArray  # (dim, mlp_hidden_dim)
    mlp_input_biases: NDArray  # (mlp_hidden_dim)
    mlp_output_weights: NDArray  # (mlp_hidden_dim, dim)
    mlp_output_biases: NDArray  # (dim)


class TextEncoderWeights(NamedTuple):
    """Weights used by the text encoder."""

    token_embedding_lut: NDArray  # (num_tokens, dim)
    """
    Lookup table mapping from token indices to dim-dimensional embeddings.
    """

    positional_encoding: NDArray  # (num_tokens, dim)
    """
    For each token position, gives a vector to offset the embedded token by to
    indicate its position.
    """

    transformer: list[ResidualAttentionBlockWeights]

    transformer_output_norm: LayerNormalisationWeights  # (dim)
    """
    Layer norm applied to the final output of the transformer.
    """

    output_projection_weights: NDArray  # (dim, dim)
    """
    The projection matrix from the transformer output of the final token to the
    embedding of the input token sequence.
    """


class Weights(NamedTuple):
    text_encoder: TextEncoderWeights


class TooManyTokensError(ValueError):
    """
    Thrown when a text string encodes to too many tokens for the current model
    to handle.
    """


def mlp(
    x: NDArray,
    input_weights: NDArray,
    input_biases: NDArray,
    output_weights: NDArray,
    output_biases: NDArray,
) -> NDArray:
    """
    A simple multi-layer perceptron with a single hidden layer and approximate
    GELU non-linearity.
    """
    x = (x @ input_weights) + input_biases
    x = approximate_gelu(x)
    x = (x @ output_weights) + output_biases
    return x


def transformer(
    x: NDArray, block_weights: list[ResidualAttentionBlockWeights]
) -> NDArray:
    """The CLIP transformer architecture."""
    for weights in block_weights:
        attention_mask = None
        if weights.attention_mask:
            attention_mask = make_attention_mask(
                x.shape[-2],
                dtype=weights.qkv_proj_weights.dtype,
            )

        # Multi-headed self attention (+ residual pass through)
        x += multi_head_attention(
            layer_normalisation(x, *weights.pre_attention_layer_norm),
            n_heads=weights.n_heads,
            qkv_proj_weights=weights.qkv_proj_weights,
            qkv_proj_biases=weights.qkv_proj_biases,
            output_proj_weights=weights.multi_head_output_proj_weights,
            output_proj_biases=weights.multi_head_output_proj_biases,
            attention_mask=attention_mask,
        )

        # Multi-layer perceptron (+ residual pass through)
        x += mlp(
            layer_normalisation(x, *weights.pre_mpl_layer_norm),
            input_weights=weights.mlp_input_weights,
            input_biases=weights.mlp_input_biases,
            output_weights=weights.mlp_output_weights,
            output_biases=weights.mlp_output_biases,
        )

    return x


def encode_text(
    texts: str | list[str], weights: TextEncoderWeights
) -> NDArray | list[NDArray]:
    """
    Encode text into the CLIP shared image/text embedding space.

    If a list of strings is given, a (string, dim) array is returned giving the
    encoding of each string in turn. Otherwise a single-dimensional vector is
    returned.
    """
    batched = True
    if isinstance(texts, str):
        batched = False
        texts = [texts]

    # Tokenise
    max_sequence_length = weights.positional_encoding.shape[0]
    tokens_padded = np.zeros((len(texts), max_sequence_length), dtype=int)
    sequence_lengths = np.empty(len(texts), dtype=int)
    for i, text in enumerate(texts):
        tokens = (
            [tokeniser.START_OF_TEXT_TOKEN]
            + list(tokeniser.encode(text))
            + [tokeniser.END_OF_TEXT_TOKEN]
        )
        if len(tokens) > max_sequence_length:
            raise TooManyTokensError(text)

        tokens_padded[i, : len(tokens)] = tokens
        sequence_lengths[i] = len(tokens)

    # Embed and positionally encode
    tokens_embedded = embedding(tokens_padded, weights.token_embedding_lut)
    tokens_embedded += weights.positional_encoding

    # To reduce computation time, strip off token positions which are never
    # used (since they have no effect on the final output due to the use of
    # causal self-attention)
    tokens_embedded = tokens_embedded[..., : np.max(sequence_lengths), :]

    # Transformer encoder
    out = transformer(tokens_embedded, weights.transformer)

    # Final layer normalisation
    out = layer_normalisation(out, *weights.transformer_output_norm)

    # Extract the output corresponding to the END_OF_TEXT_TOKEN as the final
    # embedding for each input. This is sligtly convoluted due to the need to
    # preserve arbitrary input batch dimensions.
    out = out[tuple(zip(*np.ndindex(out.shape[:-2]))) + np.s_[sequence_lengths - 1, :]]  # type: ignore

    # Apply final output projection
    out = out @ weights.output_projection_weights

    if batched:
        return out
    else:
        return out[0]
