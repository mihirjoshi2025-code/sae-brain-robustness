"""Voxelwise ridge encoding with hemodynamic delays, contiguous splits, scoring.

This is the critical-path module. Everything (dense features, SAE features, all 3
models, all datasets) plugs into `fit_score_encoding`. Kept backend-simple
(himalaya KernelRidge) so it runs on CPU.
"""
from __future__ import annotations

import numpy as np
from himalaya.ridge import RidgeCV
from himalaya.scoring import correlation_score


def add_fir_delays(X: np.ndarray, delays=(1, 2, 3, 4)) -> np.ndarray:
    """Finite-impulse-response delays to model the hemodynamic lag.

    For each delay d, shift features down by d TRs (past features predict current
    BOLD). Concatenate along the feature axis. Shape (T, F) -> (T, F*len(delays)).
    """
    T, F = X.shape
    out = np.zeros((T, F * len(delays)), dtype=X.dtype)
    for i, d in enumerate(delays):
        if d == 0:
            out[:, i * F:(i + 1) * F] = X
        else:
            out[d:, i * F:(i + 1) * F] = X[:-d]
    return out


def contiguous_split(T: int, test_frac: float = 0.2):
    """Blocked train/test split: last `test_frac` of the timeline is the test set.

    Never random-shuffle TRs (they are autocorrelated -> leakage -> inflated scores).
    """
    n_test = int(round(T * test_frac))
    train_idx = np.arange(0, T - n_test)
    test_idx = np.arange(T - n_test, T)
    return train_idx, test_idx


def zscore_fit(train: np.ndarray):
    mu = train.mean(0, keepdims=True)
    sd = train.std(0, keepdims=True)
    sd[sd == 0] = 1.0
    return mu, sd


def fit_score_encoding(
    X: np.ndarray,
    Y: np.ndarray,
    delays=(1, 2, 3, 4),
    test_frac: float = 0.2,
    alphas=np.logspace(0, 6, 10),
    seed: int = 0,
):
    """Fit ridge encoding (X features -> Y voxels) and score on held-out block.

    Returns dict: per-voxel test correlation, mean score, and shapes.
    """
    rng = np.random.default_rng(seed)  # noqa: F841 (reserved for perm tests)
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    assert X.shape[0] == Y.shape[0], f"T mismatch: {X.shape} vs {Y.shape}"

    Xd = add_fir_delays(X, delays)
    tr, te = contiguous_split(Xd.shape[0], test_frac)

    xmu, xsd = zscore_fit(Xd[tr])
    ymu, ysd = zscore_fit(Y[tr])
    Xtr, Xte = (Xd[tr] - xmu) / xsd, (Xd[te] - xmu) / xsd
    Ytr, Yte = (Y[tr] - ymu) / ysd, (Y[te] - ymu) / ysd

    model = RidgeCV(alphas=alphas)
    model.fit(Xtr, Ytr)
    pred = np.asarray(model.predict(Xte))
    scores = np.asarray(correlation_score(Yte, pred))

    return {
        "voxel_scores": scores,
        "mean_score": float(np.nanmean(scores)),
        "median_score": float(np.nanmedian(scores)),
        "n_features": int(Xd.shape[1]),
        "n_voxels": int(Y.shape[1]),
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
    }


def noise_ceiling_from_repeats(repeats: np.ndarray) -> np.ndarray:
    """Estimate per-voxel noise ceiling from repeated presentations.

    repeats: (n_rep, T, V). Returns per-voxel ceiling = mean pairwise correlation
    across repeats (a simple, honest ceiling for small data).
    """
    n_rep, T, V = repeats.shape
    cors = []
    for i in range(n_rep):
        for j in range(i + 1, n_rep):
            a, b = repeats[i], repeats[j]
            a = (a - a.mean(0)) / (a.std(0) + 1e-8)
            b = (b - b.mean(0)) / (b.std(0) + 1e-8)
            cors.append((a * b).mean(0))
    return np.mean(cors, axis=0)


def spearman_brown(r: np.ndarray, n: int) -> np.ndarray:
    """Reliability of an n-repeat average from single-repeat reliability r."""
    r = np.clip(np.nan_to_num(r), 0.0, None)
    return n * r / (1.0 + (n - 1) * r)


def noise_ceiling_corr(repeats: np.ndarray) -> dict:
    """Correlation-scale ceiling for predicting the MEAN of all repeats.

    The test target is the n-repeat average, so the single-repeat pairwise corr
    (a reliability, i.e. squared scale) understates how predictable the target is.
    Spearman-Brown gives the reliability of the n-average; its sqrt is the max
    correlation a perfect model could reach. Normalized score = corr / ceiling.
    """
    n_rep = repeats.shape[0]
    r1 = np.nan_to_num(noise_ceiling_from_repeats(repeats))
    rel_n = spearman_brown(r1, n_rep)
    return {"single_rep_r": r1, "reliability_n": rel_n,
            "ceiling": np.sqrt(rel_n), "n_rep": n_rep}
