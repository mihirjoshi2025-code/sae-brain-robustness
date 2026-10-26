"""Pereira et al. 2018 sentence fMRI (exps 2 and 3): loading, CV folds, ceiling, scoring.

Data (fetched by scripts/download_pereira.py from the paper's OSF project, osf.io/crwz7):
one `data_{384,243}sentences.mat` per participant, converted here to float32 HDF5 at
~/brain-llm-data/pereira2018/h5/{subj}_exp{2,3}.h5. Each row is ONE beta image per
sentence, already averaged over the sentence's repetitions by the authors; the
public files contain NO per-repetition or per-run data (checked: `examples_*.mat`
holds the identical matrix). Voxels are a whole-head mask in each participant's own
88x128x85 grid (not MNI), so voxel columns do not correspond across participants.

ROI / atlas maps DO ship with the data (`meta.atlases`, per participant, in that
participant's grid): `languageLH`/`languageRH` (6 fROIs each: PostTemp, AntTemp,
AngG, IFG, MFG, IFGorb), `languageParcels_aal`, `MD`, `DMN`, `visual_*`, AAL (116)
and Gordon (333). These are what Claims 3-4 need (frontal language = L_IFG,
L_IFGorb, L_MFG). The language fROIs are a few thousand voxels per hemisphere,
consistent with subject-specific localizer-defined fROIs; the files do not document
how they were derived, so treat that as unverified. (The OSF materials also ship
group probability maps, e.g. Language_overlap_n220, in MNI space; those cannot be
applied to these native-grid voxels without a registration we don't have.)

Noise ceiling: see `ceilings`. With only repetition-averaged data, the ceiling is
ACROSS-SUBJECT (other participants' responses predicting this one, Brain-Score
subject-pool extrapolation), not split-half repeats + Spearman-Brown.
"""
from __future__ import annotations

import os
from typing import NamedTuple

import h5py
import numpy as np
from himalaya.kernel_ridge import KernelRidgeCV
from himalaya.ridge import RidgeCV
from himalaya.scoring import correlation_score

ROOT = os.path.expanduser("~/brain-llm-data/pereira2018")
N_SENT = {2: 384, 3: 243}
SUBJECTS = {2: ["P01", "M02", "M04", "M07", "M08", "M09", "M14", "M15"],
            3: ["P01", "M02", "M03", "M04", "M07", "M15"]}
ALPHAS = np.logspace(-1, 9, 21)
N_OUTER, N_INNER = 8, 5
LANG_ATLASES = ("languageLH", "languageRH")


class PereiraData(NamedTuple):
    Y: np.ndarray            # (n_sent, n_vox) float32 beta per sentence
    sentences: list          # n_sent strings, as presented
    passage_ids: np.ndarray  # (n_sent,) 0-based passage index (96 or 72 passages)
    topic_ids: np.ndarray    # (n_sent,) 0-based passage category (24 per exp)
    network_masks: dict      # atlas -> (n_vox,) int ROI index, 0 = outside
    meta: dict               # roi_names, passage/topic keys, indicesIn3D, dims, task maps


def h5_path(subject: str, experiment: int) -> str:
    return f"{ROOT}/h5/{subject}_exp{experiment}.h5"


def convert(subject: str, experiment: int, prune: bool = True) -> str:
    """raw data_*.mat -> float32 HDF5 (verified), optionally deleting the .mat files."""
    import scipy.io as sio
    raw = f"{ROOT}/raw/{subject}"
    src = f"{raw}/data_{N_SENT[experiment]}sentences.mat"
    out = h5_path(subject, experiment)
    if not os.path.exists(out):
        d = sio.loadmat(src, squeeze_me=True, struct_as_record=False)
        m = d["meta"]
        Y = d["examples_passagesentences"]
        assert Y.shape[0] == N_SENT[experiment] and np.isfinite(Y).all()
        assert np.array_equal(d["labelsSentences"], np.arange(1, Y.shape[0] + 1))
        pid = d["labelsPassageForEachSentence"].astype(int) - 1
        os.makedirs(os.path.dirname(out), exist_ok=True)
        tmp = out + ".part"
        with h5py.File(tmp, "w") as h:
            h["Y"] = Y.astype(np.float32)
            h["sentences"] = np.array(list(d["keySentences"]), dtype=h5py.string_dtype())
            h["passage_ids"] = pid
            h["topic_ids"] = d["labelsPassageCategory"].astype(int)[pid] - 1
            h["passage_keys"] = np.array(list(d["keyPassages"]), dtype=h5py.string_dtype())
            h["topic_keys"] = np.array(list(d["keyPassageCategory"]), dtype=h5py.string_dtype())
            h["indicesIn3D"], h["dimensions"] = m.indicesIn3D, m.dimensions
            h["task_betas"], h["task_tstats"] = d["examples_task"], d["tstats_task"]
            h["task_labels"] = np.array(list(d["labels_task"]), dtype=h5py.string_dtype())
            g = h.create_group("atlas")
            for a, rois, mm in zip(m.atlases, m.rois, m.roiMultimask):
                ds = g.create_dataset(a, data=np.asarray(mm).astype(np.int16))
                ds.attrs["roi_names"] = [str(r) for r in np.atleast_1d(rois)]
        with h5py.File(tmp, "r") as h:  # verify before deleting anything
            assert np.array_equal(h["Y"][:], Y.astype(np.float32))
        os.rename(tmp, out)
    if prune:
        for f in (src, f"{raw}/examples_{N_SENT[experiment]}sentences.mat"):
            if os.path.exists(f):
                os.remove(f)
    return out


def load(subject: str, experiment: int) -> PereiraData:
    """One participant x experiment. See module docstring for what each field means."""
    with h5py.File(h5_path(subject, experiment), "r") as h:
        s = lambda k: [x.decode() if isinstance(x, bytes) else x for x in h[k][:]]
        masks = {a: h["atlas"][a][:] for a in h["atlas"]}
        meta = {"roi_names": {a: list(h["atlas"][a].attrs["roi_names"]) for a in h["atlas"]},
                "passage_keys": s("passage_keys"), "topic_keys": s("topic_keys"),
                "indicesIn3D": h["indicesIn3D"][:], "dimensions": h["dimensions"][:],
                "task_betas": h["task_betas"][:], "task_tstats": h["task_tstats"][:],
                "task_labels": s("task_labels")}
        return PereiraData(h["Y"][:], s("sentences"), h["passage_ids"][:],
                           h["topic_ids"][:], masks, meta)


def language_mask(d: PereiraData) -> np.ndarray:
    """Boolean (n_vox,) union of the LH and RH language fROIs."""
    return np.any([d.network_masks[a] > 0 for a in LANG_ATLASES], axis=0)


def passage_folds(passage_ids, k: int = N_OUTER) -> list:
    """[(train_idx, test_idx)] with whole passages held out (passage p -> fold p % k).

    Passages are listed topic by topic, so p % k spreads each topic's passages over
    different folds. Deterministic: no seed needed.
    """
    fold = np.asarray(passage_ids) % k
    return [(np.where(fold != f)[0], np.where(fold == f)[0]) for f in range(k)]


def _inner(passage_ids, idx):
    """Passage-grouped inner splits over the training rows `idx` (alpha selection)."""
    _, pid = np.unique(np.asarray(passage_ids)[idx], return_inverse=True)
    return passage_folds(pid, N_INNER)


def _z(train, *others):
    mu, sd = train.mean(0), train.std(0)
    sd[sd == 0] = 1.0
    return [(a - mu) / sd for a in (train, *others)]


def fold_kernels(B, passage_ids) -> list:
    """[(K_train, K_test)] linear kernels of one feature block, z-scored per fold."""
    out = []
    for tr, te in passage_folds(passage_ids):
        btr, bte = _z(np.asarray(B[tr], np.float32), np.asarray(B[te], np.float32))
        out.append((btr @ btr.T, bte @ btr.T))
    return out


def cv_predict(X, Y, passage_ids, alphas=ALPHAS, kernel: bool = False, kernels=None):
    """Out-of-fold predictions of z-scored Y (each fold z-scored on its train rows).

    X: (n, p) features; or with kernel=True a list of (n, p_s) blocks whose linear
    kernels are summed (= ridge on their concatenation, cheap when p >> n); or
    precomputed per-fold `kernels` [(K_train, K_test)] (X ignored).
    Alpha is chosen per voxel by passage-grouped inner CV on the training passages.
    Returns (pred, Yz, best_alphas_per_fold).
    """
    if kernel and kernels is None:
        ks = [fold_kernels(B, passage_ids) for B in X]
        kernels = [(sum(k[f][0] for k in ks), sum(k[f][1] for k in ks)) for f in range(N_OUTER)]
    pred, Yz, best = np.zeros(Y.shape, np.float32), np.zeros(Y.shape, np.float32), []
    for f, (tr, te) in enumerate(passage_folds(passage_ids)):
        ytr, yte = _z(Y[tr], Y[te])
        cv = _inner(passage_ids, tr)
        if kernels is not None:
            m = KernelRidgeCV(alphas=alphas, kernel="precomputed", cv=cv,
                              solver_params=dict(n_targets_batch=20000))
            m.fit(kernels[f][0], ytr)
            pred[te] = np.asarray(m.predict(kernels[f][1]))
        else:
            xtr, xte = _z(X[tr], X[te])
            m = RidgeCV(alphas=alphas, cv=cv, solver_params=dict(n_targets_batch=20000))
            m.fit(xtr, ytr)
            pred[te] = np.asarray(m.predict(xte))
        Yz[te] = yte
        best.append(np.asarray(m.best_alphas_))
    return pred, Yz, best


def fit_score(X, Y, passage_ids, **kw):
    """Per-voxel correlation between pooled out-of-fold predictions and data."""
    pred, Yz, best = cv_predict(X, Y, passage_ids, **kw)
    return np.nan_to_num(np.asarray(correlation_score(Yz, pred))), best


def _pool(ks: list, idx) -> list:
    return [(sum(ks[i][f][0] for i in idx), sum(ks[i][f][1] for i in idx)) for f in range(N_OUTER)]


def extrapolate(k, c, taus=np.logspace(-1, 1.5, 51)):
    """Fit c(k) = v0 * (1 - exp(-k / tau)) per voxel; return v0 (Brain-Score form).

    k: (K,) pool sizes; c: (K, V) mean ceiling at each size. Grid over tau with the
    least-squares v0 in closed form; v0 clipped to [0, 1]. tau is capped at ~32
    subjects so near-linear curves can't extrapolate to absurd values.
    """
    k, c = np.asarray(k, float), np.asarray(c, float)
    best_sse, v0 = np.full(c.shape[1], np.inf), np.zeros(c.shape[1])
    for t in taus:
        g = 1 - np.exp(-k / t)
        v = np.clip(g @ c / (g @ g), 0, None)
        sse = ((c - np.outer(g, v)) ** 2).sum(0)
        upd = sse < best_sse
        best_sse[upd], v0[upd] = sse[upd], v[upd]
    return np.clip(v0, 0, 1)


def ceilings(experiment: int, voxel_mask_fn=None, n_subsets: int = 3, seed: int = 0) -> dict:
    """Across-subject ceilings for every participant of an experiment (cached).

    Why this method: the public data are repetition-averaged betas (no
    per-presentation data), so split-half Spearman-Brown (used for LeBel) is
    impossible. We follow the Brain-Score Pereira ceiling (Schrimpf et al. 2021):
    the ceiling for participant s is how well OTHER participants' responses to the
    same sentences predict s's voxels, scored with exactly the model's pipeline
    (same passage-grouped outer folds, inner alpha CV, ridge on the concatenated
    other-subject voxels via summed linear kernels; voxels need not correspond
    across people because the mapping is learned). As in Brain-Score, predictors are
    only the other subjects' language-fROI voxels: with whole-brain predictors
    (~1.4M mostly-noise columns) the ceiling collapsed (P01 exp2 lang mean 0.03 at
    k=7 vs 0.07 with fROI predictors). It is the same correlation
    statistic the model gets, so no sqrt / Spearman-Brown is applied.
    Subject-pool extrapolation: for pool sizes k = 1..n-1 (up to `n_subsets`
    random pools per k, seed 0) we compute the per-voxel ceiling, average over
    pools, and fit c(k) = v0 (1 - exp(-k/tau)); v0 is the ceiling for infinitely
    many subjects. Done on the voxels picked by `voxel_mask_fn(data)` (default:
    language fROIs) because it needs ~20 fits per participant; whole-brain gets
    the k = n-1 pool ceiling only.

    Returns {subject: {"mask", "extrap" (mask voxels), "curve" (K, mask voxels),
    "k", "pool_full" (all voxels, k=n-1)}}.
    Caveats: measures only signal SHARED across people and is limited by how noisy
    the other people's data are as predictors, so it is a weak (low) ceiling: a
    768-d model can and does exceed it (normalized > 1), as GPT-2 does on Brain-
    Score's Pereira benchmark. v0 rests on a 2-parameter fit to n-1 <= 7 points
    per voxel, so it is noisy; we report the k = n-1 value too.
    """
    from itertools import combinations
    subs = SUBJECTS[experiment]
    voxel_mask_fn = voxel_mask_fn or language_mask
    path = f"{ROOT}/ceiling/exp{experiment}_ceilings.npz"
    if os.path.exists(path):
        z = np.load(path)
        return {s: {key: z[f"{s}/{key}"] for key in ("mask", "extrap", "curve", "k", "pool_full")}
                for s in subs}
    assert all(os.path.exists(h5_path(s, experiment)) for s in subs)
    pid = load(subs[0], experiment).passage_ids
    ks = []  # predictors: each other subject's mask (language fROI) voxels
    for s in subs:
        d = load(s, experiment)
        ks.append(fold_kernels(d.Y[:, voxel_mask_fn(d)], pid))
    rng, out = np.random.default_rng(seed), {}
    for i, s in enumerate(subs):
        d = load(s, experiment)
        mask = voxel_mask_fn(d)
        others = [j for j in range(len(subs)) if j != i]
        full, _ = fit_score(None, d.Y, pid, kernels=_pool(ks, others))
        kk, curve = list(range(1, len(others) + 1)), []
        for k in kk:
            pools = list(combinations(others, k))
            pick = rng.choice(len(pools), min(n_subsets, len(pools)), replace=False)
            curve.append(np.mean([fit_score(None, d.Y[:, mask], pid,
                                            kernels=_pool(ks, pools[p]))[0] for p in pick], 0))
        curve = np.stack(curve)
        out[s] = {"mask": mask, "extrap": extrapolate(kk, curve).astype(np.float32),
                  "curve": curve.astype(np.float32), "k": np.array(kk), "pool_full": full.astype(np.float32)}
        print(f"  ceiling {s} exp{experiment}: mask mean k=1 {curve[0].mean():.3f} "
              f"k={kk[-1]} {curve[-1].mean():.3f} extrap {out[s]['extrap'].mean():.3f} | "
              f"whole-brain k={kk[-1]} {full.mean():.3f}", flush=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **{f"{s}/{key}": v for s, dd in out.items() for key, v in dd.items()})
    return out
