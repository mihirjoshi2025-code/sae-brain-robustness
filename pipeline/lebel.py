"""Story-level encoding for LeBel/Huth (ds003020): shared by Gate A + layer sweep.

Word features -> TR grid (Lanczos) -> FIR delays -> z-score on train -> RidgeCV ->
per-voxel correlation. One code path so every script scores the same way.
"""
from __future__ import annotations

import os

import numpy as np
import yaml
import scipy.sparse as sp
from himalaya.kernel_ridge import KernelRidgeCV
from himalaya.ridge import RidgeCV
from himalaya.scoring import correlation_score

from pipeline import data as D
from pipeline.downsample import lanczos_downsample
from pipeline.encoding import add_fir_delays, zscore_fit

_DATA = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "configs", "data.yaml")))["podcast"]
TRAIN_STORIES = list(_DATA["train_stories"])            # 25 stories (D6, frozen 2026-10-25)
TRAIN_STORIES_ORIGINAL = list(_DATA["train_stories_original"])  # 10; GPT-2 layer sweep + Gate A
TEST_STORY = _DATA["test_story"]
ALPHAS = np.logspace(0, 6, 10)


def tr_features(wfeats: np.ndarray, story: str, n_bold: int) -> np.ndarray:
    """(N_words, d) word features -> (n_bold, d) TR-aligned features."""
    words = D.parse_textgrid_words(story)
    wtimes = np.array([(a + b) / 2.0 for _, a, b in words])  # word center time
    stim_full = lanczos_downsample(wfeats, wtimes, D.tr_times(n_bold))
    return D.trim_to_bold(stim_full, n_bold)


def story_folds(lengths: list, n_folds: int = 5) -> list:
    """Alpha-selection CV folds that hold out whole training stories.

    lengths: TRs per training story (in concatenation order). Stories are split
    into n_folds contiguous groups; returns [(train_idx, val_idx), ...] over the
    concatenated TR axis, so validation never shares a story with fitting.
    """
    edges = np.r_[0, np.cumsum(lengths)]
    groups = np.array_split(np.arange(len(lengths)), min(n_folds, len(lengths)))
    T = edges[-1]
    folds = []
    for g in groups:
        val = np.concatenate([np.arange(edges[i], edges[i + 1]) for i in g])
        folds.append((np.setdiff1d(np.arange(T), val), val))
    return folds


def _cv(cv, Xtr):
    return story_folds([x.shape[0] for x in Xtr]) if cv == "story" else cv


def fit_score(Xtr: list, Ytr: list, Xte, Yte,
              delays=(1, 2, 3, 4), alphas=ALPHAS, return_pred: bool = False,
              cv="story", info: dict | None = None):
    """Fit on concatenated train stories, return per-voxel eval correlation.

    With return_pred=True returns (scores, pred, Yte_z) for permutation tests.

    Xte/Yte may be one story's arrays or lists of stories (scored as one block).
    FIR delays are applied per story so lags never leak across story boundaries.
    cv: "story" (default; whole-story folds for alpha selection) or anything
    himalaya accepts (5 = plain contiguous KFold, used by runs before Oct 25 late).
    """
    if isinstance(Xte, np.ndarray):
        Xte, Yte = [Xte], [Yte]
    Xtr_d = np.concatenate([add_fir_delays(x, tuple(delays)) for x in Xtr])
    Ytr = np.concatenate(Ytr)
    Xte_d = np.concatenate([add_fir_delays(x, tuple(delays)) for x in Xte])
    Yte = np.concatenate(Yte)
    xmu, xsd = zscore_fit(Xtr_d)
    ymu, ysd = zscore_fit(Ytr)
    model = RidgeCV(alphas=alphas, cv=_cv(cv, Xtr))
    model.fit((Xtr_d - xmu) / xsd, (Ytr - ymu) / ysd)
    pred = np.asarray(model.predict((Xte_d - xmu) / xsd))
    if info is not None:
        info["best_alphas"] = np.asarray(model.best_alphas_)
    Yte_z = (Yte - ymu) / ysd
    scores = np.asarray(correlation_score(Yte_z, pred))
    return (scores, pred, Yte_z) if return_pred else scores


def _delay_sparse(X, delays):
    """Sparse analogue of add_fir_delays: (T, F) -> (T, F*len(delays)) CSR."""
    T, F = X.shape
    out = []
    for d in delays:
        out.append(X if d == 0 else sp.vstack([sp.csr_matrix((d, F)), X[:-d]]))
    return sp.hstack(out).tocsr()


def fit_score_kernel(Xtr: list, Ytr: list, Xte, Yte,
                     delays=(1, 2, 3, 4), alphas=ALPHAS, return_pred: bool = False,
                     info: dict | None = None, cv="story",
                     scale: str = "column", min_frac: float = 0.0):
    """Same model as fit_score (ridge on z-scored delayed features), solved in the
    dual with a precomputed linear kernel. Accepts dense or scipy-sparse features,
    so 32K-128K SAE dictionaries never get materialized as dense delayed matrices.

    With z = (x - mu) / sd and A = X diag(1/sd), m = mu / sd:
      Z1 Z2^T = A1 A2^T - (A1 m) 1^T - 1 (A2 m)^T + m.m
    Equal to fit_score on dense input up to float error (checked in BUILD_LOG).
    `info`, if given, receives best_alphas (per voxel) and n_alive (nonzero columns).
    scale="column" (default, = fit_score): unit variance per column. scale="global":
    center per column, one shared scale. min_frac > 0 drops columns nonzero in
    fewer than that fraction of training TRs. Both are sensitivity options only.
    """
    if not isinstance(Xtr, list):
        raise TypeError("Xtr must be a list of per-story matrices")
    if not isinstance(Xte, list):
        Xte, Yte = [Xte], [Yte]
    d = tuple(delays)
    Xtr_d = sp.vstack([_delay_sparse(sp.csr_matrix(x), d) for x in Xtr]).tocsr()
    Xte_d = sp.vstack([_delay_sparse(sp.csr_matrix(x), d) for x in Xte]).tocsr()
    Ytr, Yte = np.concatenate(Ytr), np.concatenate(Yte)

    mu = np.asarray(Xtr_d.mean(0)).ravel()
    sd = np.sqrt(np.maximum(np.asarray(Xtr_d.multiply(Xtr_d).mean(0)).ravel() - mu ** 2, 0))
    alive = sd > 0
    sd[~alive] = 1.0                        # dead features: centered column is all 0
    if scale == "global":
        sd = np.full_like(sd, np.sqrt(np.mean(sd[alive] ** 2)))
    w = 1.0 / sd
    if min_frac > 0:
        w[Xtr_d.getnnz(0) < min_frac * Xtr_d.shape[0]] = 0.0
        mu = np.where(w > 0, mu, 0.0)
    inv = sp.diags(w)
    Atr, Ate, m = Xtr_d @ inv, Xte_d @ inv, mu * w
    atr, ate, mm = Atr @ m, Ate @ m, float(m @ m)
    Ktr = (Atr @ Atr.T).toarray() - atr[:, None] - atr[None, :] + mm
    Kte = (Ate @ Atr.T).toarray() - ate[:, None] - atr[None, :] + mm

    ymu, ysd = zscore_fit(Ytr)
    model = KernelRidgeCV(alphas=alphas, kernel="precomputed", cv=_cv(cv, Xtr))
    model.fit(Ktr.astype(np.float64), (Ytr - ymu) / ysd)
    pred = np.asarray(model.predict(Kte.astype(np.float64)))
    if info is not None:
        info["best_alphas"] = np.asarray(model.best_alphas_)
        info["n_alive"] = int(((Xtr_d.getnnz(0) > 0) & (w > 0)).sum())
    Yte_z = (Yte - ymu) / ysd
    scores = np.asarray(correlation_score(Yte_z, pred))
    return (scores, pred, Yte_z) if return_pred else scores


def summarize(scores: np.ndarray, ceiling: np.ndarray | None = None) -> dict:
    s = np.sort(np.nan_to_num(scores))
    out = {"mean_corr": float(np.nanmean(scores)),
           "median_corr": float(np.nanmedian(scores)),
           "top1pct_corr": float(s[-max(1, len(s) // 100):].mean())}
    if ceiling is not None:
        ok = ceiling > 0
        out["mean_ceiling"] = float(ceiling.mean())
        out["normalized_mean"] = out["mean_corr"] / float(ceiling.mean())
        out["normalized_median"] = float(np.nanmedian(scores[ok] / ceiling[ok]))
    return out
