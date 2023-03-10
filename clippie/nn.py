"""
Generic deep neural network components as used by CLIP.

Only forward passes are implemented. Implementations are based on Numpy rather
than a dedicated neural network development library.
"""
from typing import TypeVar

import numpy as np

from numpy.typing import ArrayLike, NDArray, DTypeLike

from clippie.util import split_axis


T = TypeVar("T", bound=np.number)


def embedding(x: NDArray[np.integer], lut: NDArray[T]) -> NDArray[T]:
    """
    Implements an embedding layer, i.e. a simple lookup table (LUT).

    Parameters
    ==========
    x : array
        The indices to lookup in the LUT. May be of any shape.
    lut: array of shape (max_index+1, output_width)
        The lookup table for the embedding. For each index in 'x', returns the
        corresponding row of the LUT.

    Returns
    =======
    array of shape x.shape + (lut.shape[1], )
    """
    return lut[x]


def layer_normalisation(
    x: NDArray, weights: NDArray, biases: NDArray, eps: float
) -> NDArray:
    """
    Perform Layer Normalisation ("Layer Normalisation", Jimmy Lei Ba, Jamie
    Ryan Kiros, Geoffrey E.  Hinton, 2016).

    The input 'x' is assumed to be an NDArray whose shape is (...,
    num_dimensions). The values in the last dimension will be normalised to
    have a mean of 0 and a standard deviation of 1. These values are then
    scaled and offset by the weights and biases arrays respectively.

    Parameters
    ==========
    x : array (..., num_dimensions)
    weights: array (num_dimensions)
    biases: array (num_dimensions)
    eps: float
        The standard deviation is actually computed as sqrt(variance + eps)
        where eps is a small constant used to avoid division by zero.
    """
    assert x.shape[-1:] == weights.shape
    assert x.shape[-1:] == biases.shape

    # Normalise
    mean = np.mean(x, axis=-1, keepdims=True)  # (..., 1)
    variance = np.var(x, axis=-1, keepdims=True)  # (..., 1)
    x = (x - mean) / np.sqrt(variance + eps)  # (..., num_dimensions)

    # Scale and offset
    x = (x * weights) + biases

    return x


def make_attention_mask(context_length: int, dtype: DTypeLike = np.number) -> NDArray:
    """
    Produce an attention mask suitable for use in a transformer attention head
    which prevents values from paying attention to information 'in the future'.

    Produces an array of the form::

          0 -inf -inf ... -inf
          0    0 -inf ... -inf
          0    0    0 ... -inf
        ...  ...  ... ...  ...
          0    0    0 ...    0

    Params
    ======
    context_length : int
        The context length (i.e. length of the 'time' dimension)
    dtype: N
        The data type of the generated array.
    """
    all_inf = np.full((context_length, context_length), -np.inf, dtype=dtype)
    return np.triu(all_inf, k=1)


def softmax(x: NDArray, axis: int = -1) -> NDArray:
    """
    A numerically stable implementation of the soft(arg)max function.
    """
    # For reasons of numerical stability, offset all values such that we don't
    # inadvertently produce an overflow after exponentiation.
    #
    # NB: The subtraction has no effect on the outcome otherwise, e.g. because:
    #
    #     e^a / (e^a + e^b) = e^(a-x)      / (e^(a-x)      + e^(b-x))
    #                       = (e^a * e^-x) / ((e^a * e^-x) + (e^b * e^-x))
    #                       = (e^a * e^-x) / ((e^a         + e^b          ) * e^-x)
    #                       =  e^a         / ( e^a         + e^b)
    #
    # NB: We perform the subtraction locally within each axis to avoid excessively scaling down
    # unrelated values which also introduces numerical stability issues but in the opposite
    # direction.
    x = x - np.max(x, axis=axis, keepdims=True)

    x = np.exp(x)
    return x / np.sum(x, axis=axis, keepdims=True)


def sigmoid(x: NDArray) -> NDArray:
    """
    A numerically stable implementation of the sigmoid function.
    """
    # The sigmoid function is often defined as:
    #
    #     sigmoid(x) = 1 / (1 + e^(-x))
    #
    # However modestly sized negative values of X will quickly grow extremely large leading to
    # numerical precision issues and overflows. In that case, we can reformulate like so:
    #
    #     sigmoid(x) =  1  / (1   + e^(-x))
    #                = (1  / (1   + e^(-x)))     * (e^x / e^x)
    #                = e^x / (e^x + e^(-x)e^x)
    #                = e^x / (e^x + e^0)
    #                = e^x / (e^x + 1)
    #
    # In this second variation, negative values wont overflow -- though now positive values will! As
    # such, for a stable implementation we must choose the formulation according to the value of x
    # at hand.
    return np.piecewise(
        x,
        [x > 0],
        [
            # x > 0 case
            lambda x: 1.0 / (1.0 + np.exp(-x)),
            # x <= 0 case (NB slight contortion to compute np.exp(x) just once)
            lambda x: (lambda exp_x: exp_x / (exp_x + 1))(np.exp(x)),
        ],
    )


def approximate_gelu(x) -> NDArray:
    """
    An approximation of the Guassian Error Linear Unit (GELU) nonlinearity using the same approach
    and constants as the reference CLIP model.
    """
    return x * sigmoid(1.702 * x)


def multi_head_attention(
    x: NDArray,
    n_heads: int,
    qkv_proj_weights: NDArray,
    qkv_proj_biases: NDArray,
    output_proj_weights: NDArray,
    output_proj_biases: NDArray,
    attention_mask: NDArray | None = None,
):
    """
    Implements (a specific case of) multi headed (self) attention as described
    in "Attention Is All You Need" by Vaswani et al., 2017.

    Specifically performs multi-headed self attention where key, query and
    value projections are {x.shape[0]/n_heads}-dimensional and the output has
    the same dimension as the input.

    Parameters
    ==========
    x : array (..., time, dim)
        The input data giving a dim-dimensional vector for each time (a.k.a.
        position).

        Any extra dimensions (denoted by '...') will be preserved through to the output with
        multi-head self attention being applied independently to each batch.

    n_heads : int
        The number of attention heads. Must exactly divide dim.

    qkv_proj_weights : array (dim, dim*3)
        The key, query and value projections for all n_heads heads, concatenated
        into a single matrix in the order Q, K and V like so::

            +------+------+- -+------+------+------+- -+------+------+------+- -+------+
            |      |      |   |      |      |      |   |      |      |      |   |      |
            |      |      |   |      |      |      |   |      |      |      |   |      |
            | W_q0 | W_q1 |...| W_qN | W_k0 | W_k1 |...| W_kN | W_v0 | W_v1 |...| W_vN |
            |      |      |   |      |      |      |   |      |      |      |   |      |
            |      |      |   |      |      |      |   |      |      |      |   |      |
            +------+------+- -+------+------+------+- -+------+------+------+- -+------+
            |      |                 |
            |<---->|                 |
            | dim/n_heads            |
            |                        |
            |<---------------------->|
                      dim

    qkv_proj_biases : array (dim*3)
        The biases to add to the key, query and value projections for all n_heads heads,
        concatenated into a single vector in the order Q, K and V like so::

            +------+------+- -+------+------+------+- -+------+------+------+- -+------+
            | b_q0 | b_q1 |...| b_qN | b_k0 | b_k1 |...| b_kN | b_v0 | b_v1 |...| b_vN |
            +------+------+- -+------+------+------+- -+------+------+------+- -+------+
            |      |                 |
            |<---->|                 |
            | dim/n_heads            |
            |                        |
            |<---------------------->|
                      dim

    output_proj_weights : array (dim, dim)
        The matrix used to project from the concatenated attention head result values into the final
        result space.

    output_proj_biases : array (dim)
        The biases to add to the projected output values.

    attention_mask : array (time, time) or None
        If given, applies the provided attention mask to all attention heads. The value in
        attention_mask[i][j] indicates mask to apply to attention paid by time=i to time=j. Values
        should be either 0 (allow attention) or -inf (do not allow attention).

    Returns
    =======
    array (..., time, dim)
        The output of the multi-headed self attention process, applied indepdendently to each
        extraneous input dimension.
    """
    # Step 0: Project the input into its keys, queries and values
    #
    #        +-------------+
    #      t |             |
    #      i |      X      |
    #      m |             |
    #      e +-------------+
    #         dim
    #
    #          (matmul by)
    #
    #        +------+------+- -+------+------+------+- -+------+------+------+- -+------+
    #      d |      |      |   |      |      |      |   |      |      |      |   |      |
    #      i |      |      |   |      |      |      |   |      |      |      |   |      |
    #      m | W_q0 | W_q1 |...| W_qN | W_k0 | W_k1 |...| W_kN | W_v0 | W_v1 |...| W_vN |
    #        |      |      |   |      |      |      |   |      |      |      |   |      |
    #        |      |      |   |      |      |      |   |      |      |      |   |      |
    #        +------+------+- -+------+------+------+- -+------+------+------+- -+------+
    #         dim*3
    #
    #          (plus)
    #
    #        +------+------+- -+------+------+------+- -+------+------+------+- -+------+
    #        | b_q0 | b_q1 |...| b_qN | b_k0 | b_k1 |...| b_kN | b_v0 | b_v1 |...| b_vN |
    #        +------+------+- -+------+------+------+- -+------+------+------+- -+------+
    #         dim*3
    #
    #          (equals)
    #
    #        +------+------+- -+------+------+------+- -+------+------+------+- -+------+
    #      t |      |      |   |      |      |      |   |      |      |      |   |      |
    #      i |  Q0  |  Q1  |...|  QN  |  K0  |  K1  |...|  KN  |  V0  |  V1  |...|  VN  |
    #      m |      |      |   |      |      |      |   |      |      |      |   |      |
    #      e +------+------+- -+------+------+------+- -+------+------+------+- -+------+
    #         dim*3
    #
    qkv = (x @ qkv_proj_weights) + qkv_proj_biases  # (..., time, dim*3)

    # Step 1: Split into Q, K and V for each head
    #
    #           +------+          +------+          +------+
    #           |  QN  |          |  KN  |          |  VN  |
    #           ...    |          ...    |          ...    |
    #       +------+   |      +------+   |      +------+   |
    #       |  Q1  |  -+      |  K1  |  -+      |  V1  |  -+
    #     +------+ |        +------+ |        +------+ |
    #   t |      | |      t |      | |      t |      | |
    #   i |  Q0  |-+      i |  K0  |-+      i |  V0  |-+
    #   m |      |        m |      |        m |      |
    #   e +------+        e +------+        e +------+
    #      dim/n_heads       dim/n_heads       dim/n_heads
    #
    qkv = qkv.reshape(
        qkv.shape[:-1] + (-1, x.shape[-1] // n_heads)
    )  # (..., time, n_heads*3, dim//n_heads)
    qkv = np.swapaxes(qkv, -2, -3)  # (..., n_heads*3, time, dim//n_heads)
    q, k, v = np.split(
        qkv, 3, axis=-3
    )  # q, k and v are (..., n_heads, time, dim//n_heads)

    # Step 2: Lookup all queries
    #
    # The output log-attention arrays at position (,,,, i, j) give the amount of attention time 'i'
    # pays to time 'j'.
    #
    #         ...                                ...
    #       +------+           ...             +--------+
    #     +------+ |        +--------+       +--------+ |
    #   t |      | |  @ d +--------+ |  =  t |log_attn| |
    #   i |  Q0  | |    i |  K0.T  | |     i |   0    | |
    #   m |      |-+    m |        |-+     m |        |-+
    #   e +------+     /n +--------+       e +--------+
    #      dim/n_head      time               time
    #
    k_transpose = np.swapaxes(k, -1, -2)
    log_attn = q @ k_transpose  # (..., n_heads, time, time)

    # Step 3: Apply attention mask (if provided)
    if attention_mask is not None:
        log_attn += attention_mask

    # Step 4: Apply scaling to compensate for gain of matrix operation and avoid saturating the
    # softmax
    log_attn /= np.sqrt(x.shape[-1] / n_heads)

    # Step 5: Convert from log- to abosolute attention
    attn = softmax(log_attn, axis=-1)

    # Step 6: Apply attention to values
    #
    #         ...              ...               ...
    #       +--------+       +------+          +------+
    #     +--------+ |     +------+ |        +------+ |
    #   t |        | | @ t |      | |   =  t |result| |
    #   i | attn0  | |   i |  V0  | |      i |  0   | |
    #   m |        |-+   m |      |-+      m |      |-+
    #   e +--------+     e +------+        e +------+
    #      time             dim/n_head        dim/n_head
    #
    results = attn @ v  # (..., n_heads, time, dim//n_heads)

    # Step 6: Concatenate results of each attention head
    #
    #         ...
    #       +------+
    #     +------+ |          +------+------+- -+------+
    #   t |result| |        t |result|result|   |result|
    #   i |  0   | |   ==>  i |  0   |  1   |...|  N   |
    #   m |      |-+        m |      |      |   |      |
    #   e +------+          e +------+------+- -+------+
    #      dim/n_head          dim
    #
    results = np.swapaxes(results, -3, -2)  # (..., time, n_heads, dim//n_heads)
    results = results.reshape(results.shape[:-2] + (-1,))  # (..., time, dim)

    # Step 7: Apply output projection
    #
    #
    #
    #                                      +-------------+
    #     +------+------+- -+------+     d |             |
    #   t |result|result|   |result|     i |             |     +-------------+
    #   i |  0   |  1   |...|  N   |  @  m |      Wo     |  +  |     Bo      |
    #   m |      |      |   |      |       |             |     +-------------+
    #   e +------+------+- -+------+       |             |      dim
    #      dim                             |             |
    #                                      +-------------+
    #                                       dim
    #        =
    #
    #        +-------------+
    #      t |  combined   |
    #      i |   result    |
    #      m |             |
    #      e +-------------+
    #         dim
    return (results @ output_proj_weights) + output_proj_biases  # (..., time, dim)


def vit_convolve(im: NDArray, weights: NDArray) -> NDArray:
    """
    Perform the 'convolution' step of a Vision Transformer (ViT).

    .. note::

        This is not a convolution in the usual sense where a kernel is applied
        stepwise to all positions. Rather, a kernel is applied to only distinct
        (and disjoint) 'patches' -- see below.

    Based on "An Image is Worth 16x16 Words: Transformers for Image Recognition
    at Scale.", Dosovitskiy et al., 2021.

    The input image begins life as three red, green and blue images::

            B+----------+
          G+----------+ |
        R+----------+ | |
         |          | | |
         |          | |-+
         |          |-+
         +----------+

    These are then partitioned into patches of some size (e.g. 32x32 pixels)::

            B+----+----+----+
          G+----+----+----+ |
        R+----+----+----+ |-+
         |    |    |    |-+ |
         +----+----+----+ |-+
         |    |    |    |-+ |
         +----+----+----+ |-+
         |    |    |    |-+
         +----+----+----+

    For each patch, the 'weights' matrix then gives N weightings for a weighted
    sum of all pixel component values (cross-correlation). This produces an
    N-dimensional vector for each of the input patches, which is returned by
    this function.

    Parameters
    ==========
    im : array (..., h, w, 3)
        One or more RGB images.
    weights : array (dim, patch_h, patch_w, 3)
        The weights to use to perform cross-correlation on the patches of the
        image. The patch_h and patch_w dimension sizes must exactly divide the
        input image size.

    Returns
    =======
    array (..., patch, dim)
        Gives the vector computed for each image patch (in raster-scan order in
        the 'patch' dimension) for each input image.
    """

    # Determine patch sizes/counts
    patch_h = weights.shape[1]
    patch_w = weights.shape[2]

    if im.shape[-3] % patch_h != 0:
        raise ValueError(
            f"Image height ({im.shape[-3]}) not divisble by patch height ({patch_h})."
        )
    if im.shape[-2] % patch_w != 0:
        raise ValueError(
            f"Image width ({im.shape[-2]}) not divisble by patch width ({patch_w})."
        )

    num_patches_h = im.shape[-3] // patch_h
    num_patches_w = im.shape[-2] // patch_w

    # Split input image into patches
    # (num_patches_h, num_patches_w, ..., patch_h, patch_w, 3)
    im = split_axis(split_axis(im, -2, num_patches_w), -3, num_patches_h)

    # Turn each patch into a row matrix
    #
    # (num_patches_h, num_patches_w, ..., 1, pixel_index)
    #                                     \____________/
    #                                       row vector
    im = im.reshape(im.shape[:-3] + (1, patch_h * patch_w * 3))

    # Turn weights into matrix
    #
    # (pixel_index, dim)
    weights = weights.reshape(weights.shape[0], patch_h * patch_w * 3).T

    # Apply weights, performing a batched matrix multiply like so:
    #
    #   (num_patches_h, num_patches_w, ...,           1, pixel_index)
    # @                                    (pixel_index,         dim)
    # = (num_patches_h, num_patches_w, ..., 1, dim)
    im = im @ weights

    # Remove redundant axis
    #
    # (num_patches_h, num_patches_w, ..., dim)
    im = np.squeeze(im, axis=-2)

    # Combine patch axes into one
    im = np.moveaxis(im, (0, 1), (-2, -1))  # (..., dim, num_patches_h, num_patches_w)
    im = im.reshape(
        im.shape[:-2] + (num_patches_h * num_patches_w,)
    )  # (..., dim, patch)

    # Move 'dim' axis into last position, as expected for transformer
    im = np.moveaxis(im, -2, -1)  # (..., patch, dim)

    return im
