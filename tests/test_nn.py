import pytest

import numpy as np
from numpy.typing import NDArray

from clippie.nn import (
    embedding,
    layer_normalisation,
    make_attention_mask,
    softmax,
    sigmoid,
    multi_head_attention,
    vit_convolve,
)

import torch
import torch.nn


@pytest.mark.parametrize(
    "x, exp",
    [
        # Scalar
        (np.array(0), np.array([1, 10, 100, 1000])),
        # Vector
        (np.array([0, 2]), np.array([[1, 10, 100, 1000], [3, 30, 300, 3000]])),
    ],
)
def test_embedding(x: NDArray[np.integer], exp: NDArray[np.number]) -> None:
    weights = np.array(
        [
            [1, 10, 100, 1000],
            [2, 20, 200, 2000],
            [3, 30, 300, 3000],
        ]
    )

    assert np.array_equal(embedding(x, weights), exp)


def test_layer_norm() -> None:
    np.random.seed(0)

    num_dimensions = 20
    weights = np.random.uniform(size=num_dimensions)
    biases = np.random.uniform(size=num_dimensions)
    eps = 1e-5

    x = np.random.uniform(size=(2, 3, num_dimensions))

    # Use pytorch to produce a 'model' answer
    torch_layer_norm = torch.nn.LayerNorm(num_dimensions, eps=eps)
    torch_layer_norm.weight = torch.nn.Parameter(torch.tensor(weights))
    torch_layer_norm.bias = torch.nn.Parameter(torch.tensor(biases))

    exp = torch_layer_norm(torch.tensor(x)).detach().numpy()
    actual = layer_normalisation(x, weights, biases, eps)

    assert np.allclose(exp, actual)


def test_make_attention_mask() -> None:
    assert np.array_equal(
        make_attention_mask(4, dtype=np.float32),
        np.array(
            [
                [0, -np.inf, -np.inf, -np.inf],
                [0, 0, -np.inf, -np.inf],
                [0, 0, 0, -np.inf],
                [0, 0, 0, 0],
            ],
            dtype=np.float32,
        ),
    )


@pytest.mark.parametrize("dim", [0, 1, -1, -2])
def test_softmax(dim: int) -> None:
    np.random.seed(0)
    x = np.random.uniform(-10, 10, size=(3, 3))

    torch_softmax = torch.nn.Softmax(dim)
    exp = torch_softmax(torch.tensor(x)).detach().numpy()

    actual = softmax(x, axis=dim)

    print(exp)
    print(actual)

    assert np.allclose(actual, exp)


def test_sigmoid() -> None:
    x = np.arange(-100, 100, dtype=np.float32)

    exp = torch.sigmoid(torch.tensor(x)).numpy()
    actual = sigmoid(x)

    print(exp)
    print(actual)

    assert np.allclose(actual, exp)


@pytest.mark.parametrize("batch_dims", [(), (2,)])
def test_multi_head_attention(batch_dims: tuple[int]) -> None:
    np.random.seed(0)

    time = 9
    dim = 21
    n_heads = 3

    x = np.random.normal(size=batch_dims + (time, dim)).astype(np.float32)

    qkv_proj_weights = np.random.normal(size=(dim, dim * 3)).astype(np.float32)
    qkv_proj_biases = np.random.normal(size=dim * 3).astype(np.float32)
    output_proj_weights = np.random.normal(size=(dim, dim)).astype(np.float32)
    output_proj_biases = np.random.normal(size=dim).astype(np.float32)
    attn_mask = make_attention_mask(time, dtype=np.float32)

    torch_attn = torch.nn.MultiheadAttention(
        dim, n_heads, batch_first=True, dtype=torch.float32
    )
    # Insert the parameter values generated above
    torch_attn.in_proj_weight = torch.nn.Parameter(torch.tensor(qkv_proj_weights.T))
    torch_attn.in_proj_bias = torch.nn.Parameter(torch.tensor(qkv_proj_biases))
    torch_attn.out_proj.weight = torch.nn.Parameter(torch.tensor(output_proj_weights.T))
    torch_attn.out_proj.bias = torch.nn.Parameter(torch.tensor(output_proj_biases))

    x_tensor = torch.tensor(x)
    exp = torch_attn(x_tensor, x_tensor, x_tensor, attn_mask=torch.tensor(attn_mask))[0]
    exp = exp.detach().numpy()

    actual = multi_head_attention(
        x,
        n_heads=n_heads,
        qkv_proj_weights=qkv_proj_weights,
        qkv_proj_biases=qkv_proj_biases,
        output_proj_weights=output_proj_weights,
        output_proj_biases=output_proj_biases,
        attention_mask=attn_mask,
    )

    print(exp)
    print(actual)

    # NB: The relative tolerance here is pretty dire, but then we are dealing
    # with:
    #
    # * Lots of float32 arithmetic
    # * Arbitrarily scaled random weights, biases and inputs which will tend to
    #   push the softmax towards saturation and thus put more strain on the
    #   limited floating point precision
    close = np.isclose(exp, actual, rtol=1e-4)
    print(exp[~close])
    print(actual[~close])

    assert np.all(close)


def test_vit_convolve() -> None:
    np.random.seed(0)

    batch_size = 2
    dim = 20
    patch_size = 16
    patch_dim = 4
    im_size = patch_size * patch_dim

    torch_conv = torch.nn.Conv2d(
        in_channels=3,
        out_channels=dim,
        kernel_size=patch_size,
        stride=patch_size,
        bias=False,
    )

    # Extract the weights used
    weights = torch_conv.weight.detach().numpy()  # (dim, 3, h, w)
    weights = np.moveaxis(weights, 1, -1)  # (dim, h, w, 3)

    im = np.random.normal(size=(batch_size, im_size, im_size, 3)).astype(np.float32)

    # Compute answer using function under test
    actual = vit_convolve(im, weights)

    # Compute model answer using PyTorch, as used in CLIP
    im_tensor = torch.tensor(
        np.ascontiguousarray(np.moveaxis(im, -1, 1))
    )  # NB: (batch, 3, h, w)
    exp = torch_conv(im_tensor).detach().numpy()  # (batch, dim, ph, pw)
    exp = exp.reshape(batch_size, dim, patch_dim**2)  # (batch, dim, ph*pw)
    exp = np.moveaxis(exp, -2, -1)  # (batch, ph*pw, dim)

    assert np.allclose(actual, exp, atol=1e-5)
