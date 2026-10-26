"""Activation cache: one HDF5 file per model x layer x dataset x feature set.

`~/brain-llm-data/activations/{model}_L{layer}_{dataset}_{feats}.h5`, one HDF5
dataset per story holding word-level activations (N_words, d). Word level (not TR
level) so SAE encoding can reuse the same cached residual stream later. Computed
once, never recomputed (project rule).
"""
from __future__ import annotations

import os

import h5py
import numpy as np

from features.word_features import word_activations
from pipeline import data as D

ACT_ROOT = os.path.expanduser("~/brain-llm-data/activations")


def cache_path(model: str, layer: int, dataset: str, feats: str = "dense") -> str:
    return f"{ACT_ROOT}/{model.replace('/', '-')}_L{layer}_{dataset}_{feats}.h5"


def _cached(path: str, story: str):
    if not os.path.exists(path):
        return None
    with h5py.File(path, "r") as h:
        return h[story][:] if story in h else None


def word_feats(bundle_fn, model: str, story: str, layers, dataset: str = "lebel",
               feats: str = "dense"):
    """{layer: (N_words, d)} for a story, computing only the missing layers.

    `bundle_fn` is a zero-arg callable returning (model, tokenizer); it is only
    called on a cache miss so cached runs never load the model. `feats` names the
    feature set in the cache filename (e.g. "dense", "untrained") -- the caller's
    bundle_fn must return the matching model.
    """
    out, missing = {}, []
    for L in layers:
        arr = _cached(cache_path(model, L, dataset, feats), story)
        if arr is None:
            missing.append(L)
        else:
            out[L] = arr
    if missing:
        toks = [w for w, _, _ in D.parse_textgrid_words(story)]
        fresh = word_activations(bundle_fn(), toks, layer=missing)
        os.makedirs(ACT_ROOT, exist_ok=True)
        for L, arr in fresh.items():
            with h5py.File(cache_path(model, L, dataset, feats), "a") as h:
                h.create_dataset(story, data=arr.astype(np.float32))
            out[L] = arr
    return out
