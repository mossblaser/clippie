"""
Convert a PyTorch weights file into a format more convenient for clippe (and
which doesn't depend on PyTorch to load).
"""

from typing import Any

from argparse import ArgumentParser, FileType

from pathlib import Path

import pickle

import torch
import torch.jit
import torch.nn

from clippie.model import (
    Weights,
    TextEncoderWeights,
    ResidualAttentionBlockWeights,
)

TORCH_LAYER_NORM_EPS_DEFAULT = 1e-05
"""
Default value of eps used by torch.nn.LayerNorm (which is the value used by
CLIP).
"""


def extract_residual_attention_block_weights(
    module: Any,  # torch.nn.Module
    attention_mask: bool,
) -> ResidualAttentionBlockWeights:
    dim = module.attn.in_proj_bias.shape[-1] // 3

    return ResidualAttentionBlockWeights(
        pre_attention_layer_norm_weights=module.ln_1.weight.detach().numpy(),
        pre_attention_layer_norm_biases=module.ln_1.bias.detach().numpy(),
        pre_attention_layer_norm_eps=TORCH_LAYER_NORM_EPS_DEFAULT,
        n_heads=dim // 64,  # As per CLIP
        # NB: Transposed due to torch linear layers operating on transposed
        # weight matrices.
        qkv_proj_weights=module.attn.in_proj_weight.T.detach().numpy(),
        qkv_proj_biases=module.attn.in_proj_bias.detach().numpy(),
        # NB: Transposes again
        multi_head_output_proj_weights=module.attn.out_proj.weight.T.detach().numpy(),
        multi_head_output_proj_biases=module.attn.out_proj.bias.detach().numpy(),
        attention_mask=attention_mask,
        pre_mpl_layer_norm_weights=module.ln_2.weight.detach().numpy(),
        pre_mpl_layer_norm_biases=module.ln_2.bias.detach().numpy(),
        pre_mpl_layer_norm_eps=TORCH_LAYER_NORM_EPS_DEFAULT,
        # NB: Transposes again
        mlp_input_weights=module.mlp.c_fc.weight.T.detach().numpy(),
        mlp_input_biases=module.mlp.c_fc.bias.detach().numpy(),
        mlp_output_weights=module.mlp.c_proj.weight.T.detach().numpy(),
        mlp_output_biases=module.mlp.c_proj.bias.detach().numpy(),
    )


def extract_weights(
    model: Any,  # torch.nn.Module
) -> Weights:
    return Weights(
        text_encoder=TextEncoderWeights(
            token_embedding_lut=model.token_embedding.weight.detach().numpy(),
            positional_encoding=model.positional_embedding.detach().numpy(),
            transformer=[
                extract_residual_attention_block_weights(block, attention_mask=True)
                for _n, block in sorted(
                    (int(n), block)
                    for n, block in model.transformer.resblocks.named_children()
                )
            ],
            transformer_output_norm_weights=model.ln_final.weight.detach().numpy(),
            transformer_output_norm_biases=model.ln_final.bias.detach().numpy(),
            transformer_output_norm_eps=TORCH_LAYER_NORM_EPS_DEFAULT,
            output_projection_weights=model.text_projection.detach().numpy(),
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
