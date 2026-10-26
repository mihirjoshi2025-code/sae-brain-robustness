"""Contextual per-word LLM activations via sliding windows.

For each word we take the residual-stream activation at its LAST subtoken, read
from a window that gives it as much left context as possible. Layer L = the raw
output of decoder block L (blocks.L.hook_resid_post, where public SAEs attach),
read with a forward hook. Not hidden_states[L+1]: HF applies the final norm to
the last entry (GPT-2 ln_f, Gemma norm), which would mismatch the SAEs.
"""
from __future__ import annotations

import numpy as np
import torch

from features.extract import get_device


def _blocks(model):
    """Decoder block list for HF base models (GPT-2: .h, Gemma/Llama/Qwen: .layers)."""
    for name in ("h", "layers"):
        if hasattr(model, name):
            return getattr(model, name)
    raise AttributeError(f"no decoder block list on {type(model).__name__}")


@torch.no_grad()
def word_activations(bundle, words: list[str], layer,
                     max_len: int = 1024, stride: int = 512,
                     device: str | None = None):
    """words (N,) -> (N, d_model) contextual activations at `layer`.

    `layer` may be a list of layers: one forward pass, returns {layer: (N, d)}.
    """
    layers = [layer] if isinstance(layer, int) else list(layer)
    model, tok = bundle
    device = device or get_device()

    # Build token stream + map each token to its word index (space-prefixed BPE).
    tok_ids: list[int] = []
    tok_word: list[int] = []
    last_tok_of_word = {}
    for wi, w in enumerate(words):
        ids = tok(" " + w, add_special_tokens=False)["input_ids"]
        if not ids:
            ids = tok(w, add_special_tokens=False)["input_ids"]
        for t in ids:
            tok_ids.append(t)
            tok_word.append(wi)
        last_tok_of_word[wi] = len(tok_ids) - 1  # global index of word's last subtoken

    n_tok = len(tok_ids)
    d_model = model.config.hidden_size
    feats = {L: np.full((len(words), d_model), np.nan, dtype=np.float32)
             for L in layers}
    ids_arr = np.array(tok_ids)

    # Slide windows; assign a word's last-subtoken activation from the window with
    # the most left context (process left->right, later windows overwrite earlier
    # only in their fresh region).
    grab = {}
    hooks = [_blocks(model)[L].register_forward_hook(
        lambda mod, inp, out, L=L: grab.__setitem__(L, out[0] if isinstance(out, tuple) else out))
        for L in layers]
    starts = list(range(0, max(1, n_tok), stride))
    try:
        for s in starts:
            e = min(s + max_len, n_tok)
            window = torch.tensor(ids_arr[s:e][None, :], device=device)
            model(input_ids=window)
            # fresh region: for first window all, else only second half (>= s+stride)
            fresh_start = s if s == 0 else s + stride
            for L in layers:
                hs = grab[L][0].float().cpu().numpy()  # (win, d_model)
                for wi, gpos in last_tok_of_word.items():
                    if fresh_start <= gpos < e:
                        feats[L][wi] = hs[gpos - s]
            if e == n_tok:
                break
    finally:
        for h in hooks:
            h.remove()
    # Any word still NaN (edge cases): fill from nearest assigned.
    for F in feats.values():
        if np.isnan(F).any():
            good = ~np.isnan(F).any(1)
            idx = np.where(good)[0]
            for wi in np.where(~good)[0]:
                j = idx[np.argmin(np.abs(idx - wi))]
                F[wi] = F[j]
    return feats[layer] if isinstance(layer, int) else feats
