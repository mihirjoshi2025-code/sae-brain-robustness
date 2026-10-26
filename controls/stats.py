"""Significance on held-out predictions: block permutation test + FDR.

The model is never refit: we keep the test predictions, shuffle them in time in
contiguous blocks, and recompute the correlation with the real BOLD. Blocks keep
the within-block autocorrelation of the prediction intact, so the null is not
artificially narrow (shuffling single TRs would break autocorrelation and make
everything look significant).

Block length: 10 TRs = 20 s. The fitted response spans the FIR window (delays
1-4 TRs = 2-8 s) on top of a ~10-15 s hemodynamic response. On wheretheressmoke
(UTS01, top-10k voxels) the voxel-median autocorrelation of both the BOLD target
and the GPT-2 predictions drops below 0.1 by lag 5 TRs (`autocorr_length`), so
10-TR blocks are about twice that, while still giving 30 blocks on the 291-TR
test story. Null check (GPT-2 L7 predictions vs BOLD of unrelated training
stories): 5-9% of voxels at p < 0.05, 0% after FDR, p_mean 0.11-0.41.
"""
from __future__ import annotations

import numpy as np


def _z(a: np.ndarray) -> np.ndarray:
    a = a - a.mean(0)
    sd = a.std(0)
    sd[sd == 0] = 1.0
    return a / sd


def block_permutation(y: np.ndarray, pred: np.ndarray, block: int = 10,
                      n_perm: int = 1000, seed: int = 0) -> dict:
    """One-sided block-permutation test of per-voxel corr(y, pred), both (T, V).

    Returns per-voxel p, the p-value of the mean correlation over voxels, the
    observed scores, and the null distribution of the mean.
    p = (1 + #null >= observed) / (1 + n_perm).
    """
    rng = np.random.default_rng(seed)
    yz, pz = _z(np.asarray(y, np.float64)), _z(np.asarray(pred, np.float64))
    T = len(yz)
    obs = (yz * pz).mean(0)
    blocks = [np.arange(i, min(i + block, T)) for i in range(0, T, block)]
    ge = np.zeros(obs.shape, dtype=np.int64)
    null_mean = np.empty(n_perm)
    for k in range(n_perm):
        idx = np.concatenate([blocks[b] for b in rng.permutation(len(blocks))])
        null = (yz * pz[idx]).mean(0)
        ge += null >= obs
        null_mean[k] = null.mean()
    return {"p_voxel": (1 + ge) / (1 + n_perm),
            "p_mean": float((1 + (null_mean >= obs.mean()).sum()) / (1 + n_perm)),
            "obs": obs, "null_mean": null_mean}


def fdr_bh(p: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg: boolean mask of discoveries at false discovery rate q."""
    p = np.asarray(p)
    order = np.argsort(p)
    ok = p[order] <= q * np.arange(1, len(p) + 1) / len(p)
    sig = np.zeros(len(p), dtype=bool)
    if ok.any():
        sig[order[:np.flatnonzero(ok)[-1] + 1]] = True
    return sig


def autocorr_length(y: np.ndarray, thresh: float = 0.1, max_lag: int = 30) -> int:
    """Smallest lag (TRs) at which the voxel-median autocorrelation drops below thresh."""
    yz = _z(np.asarray(y, np.float64))
    for lag in range(1, max_lag + 1):
        if np.median((yz[lag:] * yz[:-lag]).mean(0)) < thresh:
            return lag
    return max_lag


def block_bootstrap_ci(y: np.ndarray, preds: list, stat_fn, block: int = 10,
                       n_boot: int = 1000, seed: int = 0, ci: float = 0.95) -> dict:
    """Block bootstrap over test TRs of a statistic of several predictions.

    y (T, V); preds: list of (T, V) predictions from different feature sets;
    stat_fn(list_of_per_voxel_corrs) -> float. TR blocks are resampled with
    replacement (blocks keep autocorrelation), per-voxel correlations recomputed.
    Returns {value, ci_low, ci_high, boot}.
    """
    rng = np.random.default_rng(seed)
    y, preds = np.asarray(y, np.float64), [np.asarray(p, np.float64) for p in preds]
    T = len(y)
    blocks = [np.arange(i, min(i + block, T)) for i in range(0, T, block)]

    def corrs(idx):
        yz = _z(y[idx])
        return [(yz * _z(p[idx])).mean(0) for p in preds]

    value = stat_fn(corrs(np.arange(T)))
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))])
        boot[b] = stat_fn(corrs(idx))
    lo, hi = np.quantile(boot, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return {"value": float(value), "ci_low": float(lo), "ci_high": float(hi), "boot": boot}
