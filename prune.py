#!/usr/bin/env python
"""
Prune LLaMA‑style models with **pluggable channel‑selection strategies**.

Supported strategies (``--strategy``):
    • ``random``  – keep channels at random (seeded)
    • ``l1``      – keep channels with largest L1 norm
    • ``l2``      – keep channels with largest L2 norm

The script measures parameter count and perplexity before and after pruning
using the reusable ``eval()`` helper.

Example:
```
python prune_llama.py \
    --model meta-llama/Llama-2-7b-hf \
    --ratio 0.5 \
    --strategy l1 \
    --dataset ptb \
    --batch_size 8
```
"""
import argparse
import random
from typing import List

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval import eval

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -----------------------------------------------------------------------------
# Channel‑selection strategies
# -----------------------------------------------------------------------------


def _importance_random(weight: torch.Tensor, keep: int, seed: int) -> List[int]:
    rng = torch.Generator(device=weight.device).manual_seed(seed)
    idx = torch.randperm(weight.size(0), generator=rng, device=weight.device)[:keep]
    return idx.sort()[0].tolist()


def _importance_norm(weight: torch.Tensor, keep: int, p: int) -> List[int]:
    scores = torch.norm(weight, p=p, dim=1)  # row‑wise norm
    _, idx = torch.topk(scores, k=keep, largest=True, sorted=True)
    return idx.tolist()


def select_channels(
    weight: torch.Tensor, ratio: float, strategy: str, seed: int
) -> List[int]:
    """Return indices of channels to **keep** according to strategy."""
    total = weight.size(0)
    keep = max(1, int(total * (1.0 - ratio)))

    strategy = strategy.lower()
    if strategy == "random":
        return _importance_random(weight, keep, seed)
    if strategy == "l1":
        return _importance_norm(weight, keep, p=1)
    if strategy == "l2":
        return _importance_norm(weight, keep, p=2)

    raise ValueError(f"Unsupported strategy: {strategy}")


# -----------------------------------------------------------------------------
# Pruning implementation
# -----------------------------------------------------------------------------


def prune_mlp_block(block: nn.Module, ratio: float, strategy: str, seed: int):
    """Prune one transformer block's MLP according to *strategy*."""
    gate_proj, up_proj, down_proj = (
        block.mlp.gate_proj,
        block.mlp.up_proj,
        block.mlp.down_proj,
    )

    # Determine which intermediate channels to keep based on gate_proj weights
    idx_keep = select_channels(gate_proj.weight, ratio, strategy, seed)
    keep = len(idx_keep)
    idx_keep_t = torch.tensor(
        idx_keep, dtype=torch.long, device=gate_proj.weight.device
    )
    dtype = gate_proj.weight.dtype

    # Slice helpers ---------------------------------------------------------
    def slice_out(layer: nn.Linear) -> nn.Linear:
        new_layer = nn.Linear(
            layer.in_features,
            keep,
            bias=layer.bias is not None,
            device=layer.weight.device,
            dtype=dtype,
        )
        new_layer.weight.data.copy_(layer.weight.data[idx_keep_t])
        if layer.bias is not None:
            new_layer.bias.data.copy_(layer.bias.data[idx_keep_t])
        return new_layer

    def slice_in(layer: nn.Linear) -> nn.Linear:
        new_layer = nn.Linear(
            keep,
            layer.out_features,
            bias=layer.bias is not None,
            device=layer.weight.device,
            dtype=dtype,
        )
        new_layer.weight.data.copy_(layer.weight.data[:, idx_keep_t])
        if layer.bias is not None:
            new_layer.bias.data.copy_(layer.bias.data)
        return new_layer

    # Replace ---------------------------------------------------------------
    block.mlp.gate_proj = slice_out(gate_proj)
    block.mlp.up_proj = slice_out(up_proj)
    block.mlp.down_proj = slice_in(down_proj)


def prune_model(
    model: nn.Module, ratio: float, strategy: str = "random", seed: int = 42
):
    for i, blk in enumerate(model.model.layers):
        prune_mlp_block(blk, ratio=ratio, strategy=strategy, seed=seed + i)
    return model


# -----------------------------------------------------------------------------
# CLI plumbing
# -----------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Channel‑prune LLaMA with multiple strategies"
    )
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument(
        "--ratio", type=float, default=0.2, help="Fraction to prune (0<ratio<1)"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="random",
        choices=["random", "l1", "l2"],
        help="Channel selection criterion",
    )
    parser.add_argument(
        "--dataset", type=str, default="wikitext2", choices=["wikitext2", "ptb", "c4"]
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_seq_len", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    print(f"Loading {args.model} …")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float32,
            device_map=None,
        )
        .to(args.device)
        .eval()
    )

    # ------------------------------ baseline
    params_before = count_parameters(model)
    ppl_before = eval(
        model,
        tokenizer,
        dataset=args.dataset,
        model_seq_len=args.model_seq_len,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(f"Params before: {params_before/1e6:.1f} M | PPL: {ppl_before:.2f}")

    # ------------------------------ pruning
    prune_model(model, ratio=args.ratio, strategy=args.strategy, seed=args.seed)

    # ------------------------------ post‑prune
    params_after = count_parameters(model)
    ppl_after = eval(
        model,
        tokenizer,
        dataset=args.dataset,
        model_seq_len=args.model_seq_len,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(f"Params after:  {params_after/1e6:.1f} M | PPL: {ppl_after:.2f}")


if __name__ == "__main__":
    main()
