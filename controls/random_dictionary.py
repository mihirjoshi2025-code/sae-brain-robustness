"""Random dictionary control (NEXT_STEPS T6): an "SAE" with no learned features.

codes = ReLU(dense @ R), R a fixed Gaussian (d_model x width) projection (seed 0);
per word keep the top-k entries, k = that word's L0 under the real SAE; then sort
columns by variance and rescale so the sorted per-feature variances match the real
SAE's. Same width, same per-word sparsity, same variance spectrum, same input
activations -- only the learned directions are missing. If real SAE features
don't beat this, the dictionary learning isn't what predicts the brain.
Variances are pooled over all stories passed in (matching statistics, not
model selection). Columns ranked past the real SAE's alive count get scale 0,
which drops ~0.06% of entries (GPT-2 L7 32k: mean L0 31.98 vs 32).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from features.extract import get_device


def _var(X: sp.csr_matrix) -> np.ndarray:
    mu = np.asarray(X.mean(0)).ravel()
    return np.maximum(np.asarray(X.multiply(X).mean(0)).ravel() - mu ** 2, 0)


@torch.no_grad()
def random_dictionary(dense: list, sae_codes: list, seed: int = 0,
                      batch: int = 512) -> list:
    """dense: per-story (N_i, d) arrays; sae_codes: matching per-story CSR codes.

    Returns per-story CSR random-dictionary codes with the SAE's width.
    """
    dev = get_device()
    width = sae_codes[0].shape[1]
    g = torch.Generator().manual_seed(seed)
    R = torch.randn(dense[0].shape[1], width, generator=g).to(dev)
    out = []
    for X, Z in zip(dense, sae_codes):
        k = np.diff(Z.indptr)                         # real SAE L0 per word
        r, c, d = [], [], []
        for i in range(0, len(X), batch):
            a = torch.relu(torch.tensor(X[i:i + batch], device=dev) @ R)
            kb = k[i:i + batch]
            v, idx = torch.topk(a, max(int(kb.max()), 1), dim=1)
            keep = np.arange(v.shape[1])[None, :] < kb[:, None]
            r.append(np.nonzero(keep)[0] + i)
            c.append(idx.cpu().numpy()[keep])
            d.append(v.cpu().numpy()[keep])
        out.append(sp.csr_matrix((np.concatenate(d), (np.concatenate(r), np.concatenate(c))),
                                 shape=(len(X), width), dtype=np.float32))

    # Match the sorted per-feature variance spectrum of the real SAE.
    v_rand = _var(sp.vstack(out).tocsr())
    v_real = np.sort(_var(sp.vstack(sae_codes).tocsr()))[::-1]
    order = np.argsort(v_rand)[::-1]
    scale = np.zeros(width)
    ok = v_rand[order] > 0
    scale[order[ok]] = np.sqrt(v_real[ok] / v_rand[order][ok])
    D = sp.diags(scale.astype(np.float32))
    return [(Z @ D).tocsr() for Z in out]
