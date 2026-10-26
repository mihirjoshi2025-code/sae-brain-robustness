"""Lanczos downsampling of word-rate features onto the TR grid.

Standard Huth-lab approach: each TR's feature is a Lanczos-weighted sum of nearby
word features (words have irregular times; TRs are a regular 2 s grid).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def lanczos_kernel(x: np.ndarray, a: int = 3) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    mask = np.abs(x) < a
    xa = x[mask]
    # sinc(x)*sinc(x/a); np.sinc already includes pi.
    out[mask] = np.sinc(xa) * np.sinc(xa / a)
    out[x == 0] = 1.0
    return out


def lanczos_matrix(word_times: np.ndarray, tr_centers: np.ndarray,
                   cutoff_mult: float = 1.0, a: int = 3) -> np.ndarray:
    """(T, N) Lanczos weights mapping word-time samples onto the TR grid.

    cutoff period = TR spacing; kernel argument scaled by that.
    """
    tr_spacing = float(np.mean(np.diff(tr_centers)))
    cutoff = cutoff_mult / tr_spacing  # frequency; period = tr_spacing
    dt = (word_times[None, :] - tr_centers[:, None]) * cutoff * 2.0
    return lanczos_kernel(dt, a=a)


def lanczos_downsample(word_feats, word_times: np.ndarray,
                       tr_centers: np.ndarray, cutoff_mult: float = 1.0,
                       a: int = 3):
    """word_feats (N, D) at word_times (N,) -> (T, D) at tr_centers (T,).

    Dense input -> dense float32; scipy sparse input (SAE codes) -> sparse CSR.
    """
    W = lanczos_matrix(np.asarray(word_times), tr_centers, cutoff_mult, a)
    if sp.issparse(word_feats):
        return sp.csr_matrix(sp.csr_matrix(W) @ word_feats, dtype=np.float32)
    return (W @ np.asarray(word_feats, dtype=np.float64)).astype(np.float32)
