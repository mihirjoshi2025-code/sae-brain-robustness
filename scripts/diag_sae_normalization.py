"""POST-HOC diagnostic (not a pre-registered result): does the Claim 1 answer depend
on how sparse features are normalized?

Motivation: under the pre-registered pipeline (z-score every column), GPT-2 SAE
features score near zero while ~80% of alive SAE columns fire in < 1% of training
TRs; z-scoring blows those rare columns up so they dominate the kernel.
Variants, fixed before running and applied identically to every feature set:
  column      pre-registered: unit variance per column (reference)
  freq1pct    drop columns nonzero in < 1% of training TRs, then unit variance
  global      center per column, one shared scale (keeps natural magnitudes)
Best frozen layer only. Rows log purpose=diagnostic_normalization.

Usage:
  python scripts/diag_sae_normalization.py --model gpt2
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from controls.random_dictionary import random_dictionary  # noqa: E402
from features.cache import word_feats  # noqa: E402
from features.sae import sae_spec, sae_word_feats  # noqa: E402
from pipeline import data as D  # noqa: E402
from pipeline import lebel as LB  # noqa: E402
from pipeline.encoding import noise_ceiling_corr  # noqa: E402
from pipeline.results import log_run  # noqa: E402

VARIANTS = {"column": {}, "freq1pct": {"min_frac": 0.01}, "global": {"scale": "global"}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    args = ap.parse_args()
    L = yaml.safe_load(open(ROOT / "configs" / "layers.yaml"))["models"][args.model]["layers"]["best"]

    reps = D.load_repeats(LB.TEST_STORY)
    nc = noise_ceiling_corr(reps)
    keep = np.argsort(nc["single_rep_r"])[::-1][:10000]
    ceil, Yte = nc["ceiling"][keep], reps.mean(0)[:, keep]
    del reps
    Ytr = [D.load_bold(s)[:, keep] for s in LB.TRAIN_STORIES]
    stories = LB.TRAIN_STORIES + [LB.TEST_STORY]
    n = dict(zip(stories, [len(y) for y in Ytr] + [len(Yte)]))

    dense = {s: word_feats(None, args.model, s, [L])[L] for s in stories}
    sets = {"dense": (dense, "", "")}
    for size in ("small", "large"):
        spec = sae_spec(args.model, size, L)
        Z = {s: sae_word_feats(args.model, L, size, s) for s in stories}
        sets[f"sae{spec['width'] // 1024}k"] = (Z, spec["width"], "")
        rd = random_dictionary([dense[s] for s in stories], [Z[s] for s in stories])
        sets[f"randdict{spec['width'] // 1024}k"] = (dict(zip(stories, rd)), spec["width"],
                                                     "random_dictionary")

    for name, (wf, width, control) in sets.items():
        X = {s: LB.tr_features(wf[s], s, n[s]) for s in stories}
        for var, kw in VARIANTS.items():
            t0 = time.time()
            info = {}
            scores = LB.fit_score_kernel([X[s] for s in LB.TRAIN_STORIES], Ytr, X[LB.TEST_STORY],
                                         Yte, info=info, **kw)
            res = LB.summarize(scores, ceil)
            print(f"{name:14s} {var:9s} mean={res['mean_corr']:.4f} norm={res['normalized_mean']:.3f} "
                  f"cols={info['n_alive']}", flush=True)
            log_run({"script": "diag_sae_normalization.py", "purpose": "diagnostic_normalization",
                     "model": args.model, "layer": L,
                     "features": "dense" if name == "dense" else "sae", "sae_size": width,
                     "control": control, "dataset": "lebel", "subject": "UTS01", "split": "test",
                     "train_stories": LB.TRAIN_STORIES, "eval_stories": LB.TEST_STORY,
                     "n_voxels": len(keep), "voxel_selection": "top single-rep r (test repeats)",
                     "delays": [1, 2, 3, 4], "alphas": "logspace(0,6,10)", "seed": 0, "cv": "story5",
                     "n_features_used": info["n_alive"], "elapsed_s": round(time.time() - t0, 1),
                     "notes": f"POST-HOC normalization variant={var}", **res})


if __name__ == "__main__":
    main()
