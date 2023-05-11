"""
A numpy based implementation of the OpenAI CLIP model's forward pass.
"""

from typing import NamedTuple, Callable

from functools import lru_cache

from pathlib import Path

from numpy.typing import NDArray

import numpy as np

from PIL import Image

import weightie

from clippie import tokeniser, __version__

from clippie.image import centre_crop_and_resize, image_to_scaled_array

from clippie.nn import (
    embedding,
    layer_normalisation,
    make_attention_mask,
    multi_head_attention,
    approximate_gelu,
    vit_convolve,
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


class ImageEncoderWeights(NamedTuple):
    """Weights used by the image encoder (a Vision Transformer (ViT))."""

    convolution_weights: NDArray  # (dim, patch_h, patch_w, 3)
    """Weights for the initial convolution in the vision transformer."""

    class_value: NDArray  # (dim)
    """
    The (static) value representing the 'class' input to the vision
    transformer.
    """

    positional_encoding: NDArray  # (patches+1, dim)
    """Positional embedding to offset each vision transformer input with."""

    pre_transformer_layer_norm: LayerNormalisationWeights  # (dim)
    """Layer norm prior to transformer."""

    transformer: list[ResidualAttentionBlockWeights]

    post_transformer_layer_norm: LayerNormalisationWeights  # (dim)
    """Layer norm following the transformer."""

    output_projection_weights: NDArray  # (dim, output_dim)
    """Final projection from transformer dimension to output dimension."""


class Weights(NamedTuple):
    text_encoder: TextEncoderWeights
    image_encoder: ImageEncoderWeights


# NB: This cache means that when load is called repeatedly and automatically by
# an encode function below it will not actually require us to re-load the
# weights file.
#
# NB: since the returned data is mmaped anyway we don't need to worry too much
# about keeping the weights loaded beyond their useful lifetime as the OS can
# swap them out if they're getting in the way anyway.
@lru_cache(maxsize=1)
def load(
    source: Path | str = "ViT-B-32.weights",
    search_paths: list[Path] | None = None,
    update: bool = False,
    progress_callback: Callable[[list[str], str, int, int | None], None]
    | None = weightie.downloader.print_status,
    min_callback_interval: float = 0.5,
) -> Weights:
    """
    Load a set of weights from a file (if a Path is given) or automatically
    download a named weights file from the GitHub release (if a string is
    given).

    Parameters
    ==========
    source : local weights file (Path) or GitHub asset filename (str)
        The weights to be downloaded.
    search_paths : [Path, ...] or None
        When the source is a GitHub asset filename, a list of locations to
        search for weights locally before resorting to downloading the file. If
        None is given, will search platform-specific data directories.

        When downloading weights, the first item on the search path (by default
        the user application data directory) will be used to store the
        downloaded weights.
    update : bool
        If True, force a check for new weights to download.
    progress_callback : f(list_of_files, file, bytes_downloaded, bytes_total) or None
        During file downloads this will be called every min_callback_interval
        seconds with the status of the download. By default this will print
        status to stderr. Disable by passing None.
    min_callback_interval : float
        See progress_callback.

    Returns
    =======
    Weights
        The model weights.

        Loaded weights are memory mmapped meaning that the data will not
        actually be read from disk until it is used and may be freely swapped
        out of RAM by the OS when needed if they're not being used.
    """
    if isinstance(source, str):
        source = weightie.download(
            repository="mossblaser/clippie",
            asset_filenames=[source],
            target_version=__version__,
            search_paths=search_paths,
            update=update,
            progress_callback=progress_callback,
            min_callback_interval=min_callback_interval,
        )[source]

    return weightie.load(source.open("rb"))


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
    texts: str | list[str], weights: TextEncoderWeights | None = None
) -> NDArray:
    """
    Encode text into the CLIP shared image/text embedding space.

    If a list of strings is given, a (len(texts), dim) array is returned giving the
    encoding of each string in turn. Otherwise a single-dimensional vector is
    returned.

    If weights are not given, a set of weights will automatically be
    downloaded using the :py:func:`load` function.
    """
    if weights is None:
        weights = load().text_encoder

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


def get_input_image_dimensions(weights: ImageEncoderWeights | None = None) -> int:
    """
    Returns the required dimensions for (square) input images, in pixels, to
    the image encoder.

    If weights are not given, a set of weights will automatically be
    downloaded using the :py:func:`load` function.
    """
    if weights is None:
        weights = load().image_encoder

    num_patches_plus_one, _ = weights.positional_encoding.shape
    num_patches = num_patches_plus_one - 1

    # NB: assume patch grid has same number of rows as cols
    patch_grid = int(num_patches**0.5)
    assert patch_grid * patch_grid == num_patches

    _, patch_h, patch_w, _ = weights.convolution_weights.shape

    image_h = patch_h * patch_grid
    image_w = patch_w * patch_grid

    assert image_h == image_w

    return image_h


def encode_image(
    images: Image.Image | list[Image.Image],
    weights: ImageEncoderWeights | None = None,
) -> NDArray | list[NDArray]:
    """
    Encode an image into the CLIP shared image/text embedding space.

    If a list of images is given, a (len(images), dim) array is returned giving the
    encoding of each string in turn. Otherwise a single-dimensional vector is
    returned.

    If weights are not given, a set of weights will automatically be
    downloaded using the :py:func:`load` function.
    """
    if weights is None:
        weights = load().image_encoder

    batched = True
    if isinstance(images, Image.Image):
        batched = False
        images = [images]

    # Load, resize and centre crop all images.
    #
    # (batch, image_dimensions, image_dimensions, 3)
    image_dimensions = get_input_image_dimensions(weights)
    image_arrays = np.stack(
        [
            image_to_scaled_array(centre_crop_and_resize(image, image_dimensions))
            for image in images
        ]
    )

    # Divide the images into patches and compute "feature" vectors from these
    #
    # (batch, patch, dim)
    image_embedded = vit_convolve(
        image_arrays.astype(weights.convolution_weights.dtype),
        weights.convolution_weights,
    )

    # Add fixed 'class' value to the set of patches
    #
    # (batch, patch+1, dim)
    transformer_input = np.empty(
        (
            image_embedded.shape[:-2]
            + (image_embedded.shape[-2] + 1, image_embedded.shape[-1])
        ),
        dtype=image_embedded.dtype,
    )
    transformer_input[..., 0, :] = weights.class_value
    transformer_input[..., 1:, :] = image_embedded

    # Add positional encoding
    transformer_input += weights.positional_encoding

    # Pre-transformer layer normalisation
    transformer_input = layer_normalisation(
        transformer_input, *weights.pre_transformer_layer_norm
    )

    # Transformer
    transformer_output = transformer(transformer_input, weights.transformer)

    # Extract the output value corresponding to the 'class' input.
    #
    # (batch, dim)
    out = transformer_output[..., 0, :]

    # Post-transformer layer normalisation
    out = layer_normalisation(out, *weights.post_transformer_layer_norm)

    # Finally, project to the final output space (typically a dimensional
    # reduction)
    #
    # (batch, output_dim)
    out = out @ weights.output_projection_weights

    if batched:
        return out
    else:
        return out[0]
