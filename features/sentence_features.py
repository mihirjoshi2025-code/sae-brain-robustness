"""One activation vector per sentence (Pereira 2018), cached per model x layer.

Decision (D8, fixed before any Pereira score was computed): the feature for a
sentence is the residual-stream activation at its LAST token, read while the model
sees the preceding sentences of the same passage as context. Why last-token, not
mean-pool:
- GPT-2 is causal, so only the last position has attended to the whole sentence;
  a mean-pool is dominated by early positions that have seen only a prefix.
- It matches the word-level convention (last subtoken per word) and how SAE
  features will be read later, so dense and SAE inputs are the same vectors.
- Participants read each passage's sentences in order, so passage context is the
  ecologically matched input. Context never crosses passages.
The last token is usually the sentence-final period. Each passage is prefixed with
BOS (`<|endoftext|>`), whose position is dropped; this avoids the first-position
attention-sink artifact for short contexts and matches GPT-2 SAE training.
Layer L = raw output of decoder block L (forward hook, pre final norm), as in
`features.word_features`.
"""
from __future__ import annotations

import os

import h5py
import numpy as np
import torch

from features.cache import ACT_ROOT, cache_path
from features.extract import get_device
from features.word_features import _blocks


@torch.no_grad()
def sentence_activations(bundle, sentences: list[str], passage_ids, layers,
                         device: str | None = None) -> dict:
    """sentences (N,), passage_ids (N,) -> {layer: (N, d_model)} last-token acts."""
    model, tok = bundle
    device = device or get_device()
    passage_ids = np.asarray(passage_ids)
    d = model.config.hidden_size
    out = {L: np.zeros((len(sentences), d), np.float32) for L in layers}
    grab = {}
    hooks = [_blocks(model)[L].register_forward_hook(
        lambda m, i, o, L=L: grab.__setitem__(L, o[0] if isinstance(o, tuple) else o))
        for L in layers]
    try:
        for p in dict.fromkeys(passage_ids.tolist()):  # passages in stimulus order
            idx = np.where(passage_ids == p)[0]
            ids, ends = [tok.bos_token_id], []
            for i in idx:
                ids += tok(" " + sentences[i].strip(), add_special_tokens=False)["input_ids"]
                ends.append(len(ids) - 1)
            model(input_ids=torch.tensor([ids], device=device))
            for L in layers:
                out[L][idx] = grab[L][0, ends].float().cpu().numpy()
    finally:
        for h in hooks:
            h.remove()
    return out


def sentence_feats(bundle_fn, model: str, key: str, sentences, passage_ids, layers,
                   dataset: str = "pereira", feats: str = "dense") -> dict:
    """Cached {layer: (N, d)}; one HDF5 dataset per stimulus set `key` (e.g. 'exp2').

    `bundle_fn` is only called on a cache miss. The sentence list is stored as an
    attribute and checked on read so a stale cache can't be silently reused.
    """
    out, missing = {}, []
    for L in layers:
        p = cache_path(model, L, dataset, feats)
        if os.path.exists(p):
            with h5py.File(p, "r") as h:
                if key in h:
                    assert h[key].attrs["n_sentences"] == len(sentences)
                    assert h[key].attrs["first"] == sentences[0]
                    out[L] = h[key][:]
                    continue
        missing.append(L)
    if missing:
        fresh = sentence_activations(bundle_fn(), list(sentences), passage_ids, missing)
        os.makedirs(ACT_ROOT, exist_ok=True)
        for L, arr in fresh.items():
            with h5py.File(cache_path(model, L, dataset, feats), "a") as h:
                ds = h.create_dataset(key, data=arr.astype(np.float32))
                ds.attrs.update(n_sentences=len(sentences), first=sentences[0],
                                readout="last_token_passage_context_bos")
            out[L] = arr
    return out
