"""Extract dense LLM activations for a list of text chunks.

Uses HuggingFace `transformers` with output_hidden_states (stable across versions).
hidden_states[L+1] is the residual stream after block L -- the same place public
SAEs (Gemma Scope / Llama Scope / GPT-2 SAEs) attach, so SAE extraction will reuse
this hook point later.

For the smoke test we aggregate per-chunk (mean over tokens); the real pipeline
aligns tokens -> words -> TRs.
"""
from __future__ import annotations

import numpy as np
import torch


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(model_name: str = "gpt2", device: str | None = None):
    from transformers import AutoModel, AutoTokenizer

    device = device or get_device()
    tok = AutoTokenizer.from_pretrained(model_name)
    kw = {}
    if "gemma" in model_name.lower():
        # >1B params: bf16 weights (activations are cast to float32 downstream);
        # Gemma 2 needs eager attention for its logit softcapping.
        kw["dtype"] = torch.bfloat16
        if "gemma-2" in model_name.lower():
            kw["attn_implementation"] = "eager"
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True, **kw)
    model.to(device).eval()
    return model, tok


@torch.no_grad()
def dense_activations(bundle, chunks: list[str], layer: int, device: str | None = None):
    """Return (n_chunks, d_model) mean-pooled residual-stream activations at `layer`.

    `bundle` is (model, tokenizer) from load_model. `layer` is the block index;
    we read hidden_states[layer + 1] = residual stream after that block.
    """
    model, tok = bundle
    device = device or get_device()
    out = []
    for text in chunks:
        enc = tok(text, return_tensors="pt").to(device)
        hs = model(**enc).hidden_states  # tuple len n_layers+1
        acts = hs[layer + 1][0]          # (seq, d_model)
        out.append(acts.mean(0).float().cpu().numpy())
    return np.stack(out).astype(np.float32)
