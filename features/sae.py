"""SAE feature codes from cached dense word activations (SAELens public SAEs).

Codes are sparse (TopK / JumpReLU), so they are cached as CSR in one HDF5 file per
model x layer x dataset x SAE size (`..._sae{width}.h5`, a group per story), next
to the dense cache. Computed once from the dense cache, never recomputed.
"""
from __future__ import annotations

import os

import h5py
import numpy as np
import scipy.sparse as sp
import torch
import yaml

from features.cache import cache_path
from features.extract import get_device

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sae_spec(model: str, size: str, layer: int) -> dict:
    """{release, sae_id, width} for model/size ('small'|'large') from configs/models.yaml."""
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "models.yaml")))
    s = dict(cfg["models"][model]["saes"][size])
    s["sae_id"] = s["sae_id"].format(L=layer)
    return s


def load_sae(release: str, sae_id: str, device: str | None = None):
    from sae_lens import SAE
    sae = SAE.from_pretrained(release, sae_id, device=device or get_device())
    return (sae[0] if isinstance(sae, tuple) else sae).eval()


@torch.no_grad()
def encode_sparse(sae, X: np.ndarray, batch: int = 512) -> tuple[sp.csr_matrix, float]:
    """(N, d_in) dense -> (N, d_sae) CSR codes, plus reconstruction FVU."""
    dev = next(sae.parameters()).device
    blocks, sse = [], 0.0
    for i in range(0, len(X), batch):
        x = torch.tensor(X[i:i + batch], device=dev)
        z = sae.encode(x)
        sse += float(((x - sae.decode(z)) ** 2).sum())
        blocks.append(sp.csr_matrix(z.float().cpu().numpy()))
    sst = float(((X - X.mean(0)) ** 2).sum())
    return sp.vstack(blocks).tocsr().astype(np.float32), sse / sst


def sae_word_feats(model: str, layer: int, size: str, story: str,
                   dataset: str = "lebel", sae_cache: dict | None = None) -> sp.csr_matrix:
    """Cached CSR SAE codes (N_words, d_sae) for a story; encodes on a miss."""
    spec = sae_spec(model, size, layer)
    path = cache_path(model, layer, dataset, f"sae{spec['width'] // 1024}k")
    if os.path.exists(path):
        with h5py.File(path, "r") as h:
            if story in h:
                g = h[story]
                return sp.csr_matrix((g["data"][:], g["indices"][:], g["indptr"][:]),
                                     shape=tuple(g.attrs["shape"]))
    with h5py.File(cache_path(model, layer, dataset), "r") as h:
        X = h[story][:]
    key = (spec["release"], spec["sae_id"])
    if sae_cache is None:
        sae_cache = {}
    if key not in sae_cache:
        sae_cache[key] = load_sae(*key)
    Z, fvu = encode_sparse(sae_cache[key], X)
    with h5py.File(path, "a") as h:
        g = h.create_group(story)
        for k in ("data", "indices", "indptr"):
            g.create_dataset(k, data=getattr(Z, k), compression="gzip")
        g.attrs["shape"] = Z.shape
        g.attrs["fvu"] = fvu
        g.attrs["release"], g.attrs["sae_id"] = key
    return Z


def sae_fvu(model: str, layer: int, size: str, dataset: str = "lebel") -> float:
    """Mean reconstruction FVU over cached stories (quality check for the write-up)."""
    spec = sae_spec(model, size, layer)
    with h5py.File(cache_path(model, layer, dataset, f"sae{spec['width'] // 1024}k"), "r") as h:
        return float(np.mean([h[s].attrs["fvu"] for s in h]))
