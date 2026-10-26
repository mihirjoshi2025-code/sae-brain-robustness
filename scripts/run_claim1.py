"""Claim 1 (NEXT_STEPS T5): do sparse SAE features predict the brain about as well
as dense ones?

Per frozen layer (configs/layers.yaml): fit the SAME encoding model (ridge on
z-scored FIR-delayed features, kernel solver, story-level alpha folds) on dense
activations, both SAE sizes (configs/models.yaml), and the random-dictionary
control for each size; test on `wheretheressmoke`. Every row gets a block-
permutation p-value and FDR count. SAE rows carry a 10-TR block-bootstrap 95% CI
of ratio = normalized_mean_SAE / normalized_mean_dense (ceiling cancels) and the
pre-registered verdict: holds if ci_low >= 0.9, fails if ci_high < 0.9, else
inconclusive. SAE quality (FVU, L0) is logged as purpose=sae_check.

Usage:
  python scripts/run_claim1.py --model gpt2
"""
import argparse
import resource
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from controls import stats as S  # noqa: E402
from controls.random_dictionary import random_dictionary  # noqa: E402
from features.cache import word_feats  # noqa: E402
from features.extract import load_model  # noqa: E402
from features.sae import sae_fvu, sae_spec, sae_word_feats  # noqa: E402
from pipeline import data as D  # noqa: E402
from pipeline import lebel as LB  # noqa: E402
from pipeline.encoding import noise_ceiling_corr  # noqa: E402
from pipeline.results import log_run  # noqa: E402

SEED = 0
THRESH = 0.9


def verdict(lo, hi):
    return "holds" if lo >= THRESH else "fails" if hi < THRESH else "inconclusive"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--subject", default="UTS01")
    ap.add_argument("--voxels", type=int, default=10000)
    ap.add_argument("--delays", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--layers", type=int, nargs="+", help="subset of frozen layers")
    args = ap.parse_args()
    np.random.seed(SEED)

    cfg = yaml.safe_load(open(ROOT / "configs" / "layers.yaml"))["models"][args.model]
    layers = args.layers or sorted(set(cfg["layers"].values()))

    reps = D.load_repeats(LB.TEST_STORY, args.subject)
    nc = noise_ceiling_corr(reps)
    keep = np.argsort(nc["single_rep_r"])[::-1][:args.voxels]
    ceil = nc["ceiling"][keep]
    Yte = reps.mean(0)[:, keep]
    del reps
    Ytr = [D.load_bold(st, args.subject)[:, keep] for st in LB.TRAIN_STORIES]
    stories = LB.TRAIN_STORIES + [LB.TEST_STORY]
    n_tr = dict(zip(stories, [len(y) for y in Ytr] + [len(Yte)]))
    base = {"model": args.model, "dataset": "lebel", "subject": args.subject,
            "train_stories": LB.TRAIN_STORIES, "eval_stories": LB.TEST_STORY,
            "n_voxels": len(keep), "voxel_selection": "top single-rep r (test repeats)",
            "delays": args.delays, "alphas": "logspace(0,6,10)", "seed": SEED,
            "cv": "story5"}
    saes = {}

    def score(name, wf, L, row):
        t0 = time.time()
        X = {s: LB.tr_features(wf[s], s, n_tr[s]) for s in stories}
        info = {}
        scores, pred, yz = LB.fit_score_kernel([X[s] for s in LB.TRAIN_STORIES], Ytr,
                                               X[LB.TEST_STORY], Yte, delays=args.delays,
                                               return_pred=True, info=info)
        res = LB.summarize(scores, ceil)
        perm = S.block_permutation(yz, pred, n_perm=1000, seed=SEED)
        sig = S.fdr_bh(perm["p_voxel"])
        at_max = float(np.mean(info["best_alphas"] >= LB.ALPHAS.max()))
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30
        row = {**base, "layer": L, "split": "test", "p_value": perm["p_mean"],
               "n_perm": 1000, "frac_voxels_sig": float(sig.mean()),
               "n_sig_voxels_fdr": int(sig.sum()), "n_features_used": info["n_alive"],
               "elapsed_s": round(time.time() - t0, 1), **res, **row}
        row["notes"] = (f"kernel solver; frac_voxels_alpha_at_grid_max={at_max:.3f}; "
                        f"peak_rss_gb={rss:.2f}; " + row.get("notes", "")).rstrip("; ")
        print(f"  L{L:2d} {name:16s} mean={res['mean_corr']:.4f} norm={res['normalized_mean']:.3f} "
              f"sig={sig.mean():.2f} p={perm['p_mean']:.3f} alive={info['n_alive']} "
              f"({row['elapsed_s']:.0f}s)", flush=True)
        return row, pred, yz

    for L in layers:
        print(f"layer {L}", flush=True)
        dense = {s: word_feats(lambda: load_model(args.model), args.model, s, [L])[L]
                 for s in stories}
        drow, dpred, yz = score("dense", dense, L, {"purpose": "claim1_sae_vs_dense",
                                                    "features": "dense"})
        log_run(drow)
        for size in ("small", "large"):
            spec = sae_spec(args.model, size, L)
            Z = {s: sae_word_feats(args.model, L, size, s, sae_cache=saes) for s in stories}
            l0 = float(np.mean(np.concatenate([np.diff(Z[s].indptr) for s in stories])))
            fvu = sae_fvu(args.model, L, size)
            log_run({**base, "purpose": "sae_check", "layer": L, "features": "sae",
                     "sae_size": spec["width"], "sae_release": spec["release"],
                     "split": "all", "eval_stories": stories, "train_stories": "",
                     "notes": f"fvu={fvu:.4f}; mean_L0={l0:.1f}; "
                              f"pass_fvu<0.4={fvu < 0.4}; sae_id={spec['sae_id']}"})

            srow, spred, _ = score(f"sae {spec['width'] // 1024}k", Z, L, {
                "purpose": "claim1_sae_vs_dense", "features": "sae",
                "sae_size": spec["width"], "sae_release": spec["release"],
                "notes": f"sae_fvu={fvu:.3f}"})
            ratio = S.block_bootstrap_ci(yz, [spred, dpred], lambda c: c[0].mean() / c[1].mean(),
                                         seed=SEED)
            diff = S.block_bootstrap_ci(yz, [spred, dpred], lambda c: (c[0] - c[1]).mean(),
                                        seed=SEED)
            v = verdict(ratio["ci_low"], ratio["ci_high"])
            srow.update(ci_low=ratio["ci_low"], ci_high=ratio["ci_high"])
            srow["notes"] += (f"; ratio_vs_dense={ratio['value']:.3f} "
                              f"[{ratio['ci_low']:.3f},{ratio['ci_high']:.3f}]; "
                              f"diff_corr={diff['value']:.4f} [{diff['ci_low']:.4f},"
                              f"{diff['ci_high']:.4f}]; claim1_verdict={v}")
            log_run(srow)
            print(f"    ratio SAE/dense = {ratio['value']:.3f} [{ratio['ci_low']:.3f}, "
                  f"{ratio['ci_high']:.3f}] -> {v}", flush=True)

            rd = random_dictionary([dense[s] for s in stories], [Z[s] for s in stories], seed=SEED)
            rrow, rpred, _ = score(f"randdict {spec['width'] // 1024}k",
                                   dict(zip(stories, rd)), L, {
                                       "purpose": "claim1_sae_vs_dense", "features": "sae",
                                       "control": "random_dictionary",
                                       "sae_size": spec["width"], "sae_release": spec["release"]})
            vs = S.block_bootstrap_ci(yz, [spred, rpred], lambda c: (c[0] - c[1]).mean(), seed=SEED)
            rrow["notes"] += (f"; sae_minus_randdict_corr={vs['value']:.4f} "
                              f"[{vs['ci_low']:.4f},{vs['ci_high']:.4f}]")
            log_run(rrow)


if __name__ == "__main__":
    main()
