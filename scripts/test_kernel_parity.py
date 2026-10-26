"""Parity test (NEXT_STEPS T4): kernel solver == primal ridge on dense features.

GPT-2 L9 dense, top-10k voxels, same story-level alpha folds for both. Passes
if voxels that pick the same alpha agree to |delta corr| < 1e-3 and at most 0.1%
of voxels flip alpha (near-tied CV scores flip on float noise: on L9, 1/10000
voxel moves one grid step). Prints peak RSS.

Usage:
  python scripts/test_kernel_parity.py [--layer 9]
"""
import argparse
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.cache import word_feats  # noqa: E402
from features.extract import load_model  # noqa: E402
from pipeline import data as D  # noqa: E402
from pipeline import lebel as LB  # noqa: E402
from pipeline.encoding import noise_ceiling_corr  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=9)
    args = ap.parse_args()
    L = args.layer

    reps = D.load_repeats(LB.TEST_STORY)
    keep = np.argsort(noise_ceiling_corr(reps)["single_rep_r"])[::-1][:10000]
    Yte = reps.mean(0)[:, keep]
    del reps
    Ytr = [D.load_bold(s)[:, keep] for s in LB.TRAIN_STORIES]
    stories = LB.TRAIN_STORIES + [LB.TEST_STORY]
    n = dict(zip(stories, [len(y) for y in Ytr] + [len(Yte)]))
    X = {s: LB.tr_features(word_feats(lambda: load_model("gpt2"), "gpt2", s, [L])[L], s, n[s])
         for s in stories}
    Xtr = [X[s] for s in LB.TRAIN_STORIES]

    ia, ib = {}, {}
    t = time.time(); a = LB.fit_score(Xtr, Ytr, X[LB.TEST_STORY], Yte, info=ia); ta = time.time() - t
    t = time.time(); b = LB.fit_score_kernel(Xtr, Ytr, X[LB.TEST_STORY], Yte, info=ib); tb = time.time() - t
    same = np.isclose(ia["best_alphas"], ib["best_alphas"])
    d_same = float(np.abs(a - b)[same].max())
    flip = float(1 - same.mean())
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30  # bytes on macOS
    print(f"layer {L}: primal mean {a.mean():.5f} ({ta:.0f}s) | kernel mean {b.mean():.5f} ({tb:.0f}s)")
    print(f"  same-alpha max|dcorr| {d_same:.2e} | alpha flips {int((~same).sum())}/{len(same)} "
          f"| overall max|dcorr| {np.abs(a - b).max():.2e} | peak RSS {rss:.2f} GB")
    ok = d_same < 1e-3 and flip <= 1e-3
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
