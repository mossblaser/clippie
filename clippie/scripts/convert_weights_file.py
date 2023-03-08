"""
Convert a PyTorch weights file into a format more convenient for clippe (and
which doesn't depend on PyTorch to load).
"""

from typing import Any

from argparse import ArgumentParser, FileType

from pathlib import Path

import pickle

import numpy as np
from numpy.typing import NDArray

import torch
import torch.jit
import torch.nn

from clippie.model import (
    Weights,
    TextEncoderWeights,
    ImageEncoderWeights,
    ResidualAttentionBlockWeights,
    LayerNormalisationWeights,
)

TORCH_LAYER_NORM_EPS_DEFAULT = 1e-05
"""
Default value of eps used by torch.nn.LayerNorm (which is the value used by
CLIP).
"""

def to_np(tensor: torch.Tensor) -> NDArray:
    return np.ascontiguousarray(tensor.detach().numpy()).astype(np.float32)


def extract_residual_attention_block_weights(
    module: Any,  # torch.nn.Module
    attention_mask: bool,
) -> ResidualAttentionBlockWeights:
    dim = module.attn.in_proj_bias.shape[-1] // 3

    return ResidualAttentionBlockWeights(
        pre_attention_layer_norm=LayerNormalisationWeights(
            weights=to_np(module.ln_1.weight),
            biases=to_np(module.ln_1.bias),
            eps=TORCH_LAYER_NORM_EPS_DEFAULT,
        ),
        n_heads=dim // 64,  # As per CLIP
        # NB: Transposed due to torch linear layers operating on transposed
        # weight matrices.
        qkv_proj_weights=to_np(module.attn.in_proj_weight.T),
        qkv_proj_biases=to_np(module.attn.in_proj_bias),
        # NB: Transposes again
        multi_head_output_proj_weights=to_np(module.attn.out_proj.weight.T),
        multi_head_output_proj_biases=to_np(module.attn.out_proj.bias),
        attention_mask=attention_mask,
        pre_mpl_layer_norm=LayerNormalisationWeights(
            weights=to_np(module.ln_2.weight),
            biases=to_np(module.ln_2.bias),
            eps=TORCH_LAYER_NORM_EPS_DEFAULT,
        ),
        # NB: Transposes again
        mlp_input_weights=to_np(module.mlp.c_fc.weight.T),
        mlp_input_biases=to_np(module.mlp.c_fc.bias),
        mlp_output_weights=to_np(module.mlp.c_proj.weight.T),
        mlp_output_biases=to_np(module.mlp.c_proj.bias),
    )


def extract_transformer_weights(
    transformer: Any,  # nn.Module (specifically clip.model.Transformer)
    attention_mask: bool,
) -> list[ResidualAttentionBlockWeights]:
    return [
        extract_residual_attention_block_weights(block, attention_mask=attention_mask)
        for _n, block in sorted(
            (int(n), block)
            for n, block in transformer.resblocks.named_children()
        )
    ]


def extract_weights(
    model: Any,  # torch.nn.Module
) -> Weights:
    return Weights(
        text_encoder=TextEncoderWeights(
            token_embedding_lut=to_np(model.token_embedding.weight),
            positional_encoding=to_np(model.positional_embedding),
            transformer=extract_transformer_weights(model.transformer, attention_mask=True),
            transformer_output_norm=LayerNormalisationWeights(
                weights=to_np(model.ln_final.weight),
                biases=to_np(model.ln_final.bias),
                eps=TORCH_LAYER_NORM_EPS_DEFAULT,
            ),
            output_projection_weights=to_np(model.text_projection),
        ),
    )


def main():
    parser = ArgumentParser(
        "Convert a CLIP weights file into the format used by clippie."
    )
    parser.add_argument("input", type=FileType("rb"))
    parser.add_argument("output", type=FileType("wb"))
    args = parser.parse_args()

    # To avoid warnings, attempt to open as a PyTorch JIT module before falling
    # back on regular module loader.
    try:
        model = torch.jit.load(args.input, map_location="cpu")
    except RuntimeError:
        args.input.seek(0)
        model = torch.load(args.input, map_location="cpu")

    pickle.dump(extract_weights(model), args.output)


if __name__ == "__main__":
    main()
