"""Untrained-model baseline: same architecture, random weights (fixed seed).

`AutoModel.from_config` with `torch.manual_seed(seed)` gives the model's own
default init (GPT-2: N(0, 0.02) weights) with no training. Word activations go
through the exact same path as the trained model (`features.cache.word_feats`,
last subtoken, 1024/512 windows, hidden_states[L+1]) and are cached as
`{model}_L{layer}_{dataset}_untrained.h5`. Dense only (no SAE exists for it).
"""
from __future__ import annotations

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from features.cache import word_feats
from features.extract import get_device


def load_untrained(model_name: str = "gpt2", seed: int = 0, device: str | None = None):
    """(model, tokenizer) with random init; same seed -> identical weights."""
    device = device or get_device()
    cfg = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
    torch.manual_seed(seed)
    model = AutoModel.from_config(cfg).to(device).eval()
    return model, AutoTokenizer.from_pretrained(model_name)


def feats_name(seed: int = 0) -> str:
    return "untrained" if seed == 0 else f"untrained-s{seed}"


def provider(model_name: str, layer: int, layers=None, seed: int = 0,
             dataset: str = "lebel"):
    """story -> (N_words, d) untrained activations at `layer` (cached).

    `layers`: every layer this run needs, so a cache miss fills them in one pass.
    """
    layers = sorted(set(layers or []) | {layer})
    _b = []

    def bundle():
        if not _b:
            print(f"loading untrained {model_name} (seed {seed})...")
            _b.append(load_untrained(model_name, seed))
        return _b[0]

    return lambda story: word_feats(bundle, model_name, story, layers, dataset,
                                    feats=feats_name(seed))[layer]
