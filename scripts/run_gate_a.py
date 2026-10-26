"""Gate A: dense-feature brain-encoding baseline on real fMRI.

Train an encoding model (GPT-2 dense features -> BOLD) on the training stories,
test on `wheretheressmoke` (mean of its repeats), and score against the
Spearman-Brown-corrected noise ceiling. Every run appends a row to
`results/runs.csv`.

Usage:
  python scripts/run_gate_a.py --layer 9
  python scripts/run_gate_a.py --frozen     # all layers frozen in configs/layers.yaml
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.cache import word_feats  # noqa: E402
from features.extract import load_model  # noqa: E402
from pipeline import data as D  # noqa: E402
from pipeline import lebel as LB  # noqa: E402
from pipeline.encoding import noise_ceiling_corr  # noqa: E402
from pipeline.results import log_run  # noqa: E402

SEED = 0


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--layer", type=int, nargs="+", default=[9])
    g.add_argument("--frozen", action="store_true",
                   help="score the layers frozen in configs/layers.yaml")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--train", nargs="+", default=LB.TRAIN_STORIES)
    ap.add_argument("--subject", default="UTS01")
    ap.add_argument("--voxels", type=int, default=10000,
                    help="keep top-N voxels by noise ceiling (speed/memory)")
    ap.add_argument("--delays", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--notes", default="")
    args = ap.parse_args()
    np.random.seed(SEED)

    if args.frozen:
        cfg = yaml.safe_load(open(ROOT / "configs" / "layers.yaml"))
        layers = sorted(set(cfg["models"][args.model]["layers"].values()))
    else:
        layers = args.layer

    _bundle = []
    def bundle():  # load the model only on an activation-cache miss
        if not _bundle:
            print(f"loading {args.model}...")
            _bundle.append(load_model(args.model))
        return _bundle[0]

    # --- noise ceiling from test-story repeats -> also picks voxel subset ---
    reps = D.load_repeats(LB.TEST_STORY, args.subject)    # (n_rep, T, V)
    nc = noise_ceiling_corr(reps)
    keep = np.argsort(nc["single_rep_r"])[::-1][:args.voxels]
    ceil = nc["ceiling"][keep]
    print(f"kept {len(keep)} voxels; single-rep r mean={nc['single_rep_r'][keep].mean():.3f}; "
          f"SB({nc['n_rep']}) ceiling mean={ceil.mean():.3f}")

    Ytr = [D.load_bold(st, args.subject)[:, keep] for st in args.train]
    Yte = reps.mean(0)[:, keep]
    stories = args.train + [LB.TEST_STORY]
    wf = {st: word_feats(bundle, args.model, st, layers) for st in stories}

    for L in layers:
        t0 = time.time()
        Xtr = [LB.tr_features(wf[st][L], st, len(y)) for st, y in zip(args.train, Ytr)]
        Xte = LB.tr_features(wf[LB.TEST_STORY][L], LB.TEST_STORY, len(Yte))
        scores = LB.fit_score(Xtr, Ytr, Xte, Yte, delays=args.delays)
        res = LB.summarize(scores, ceil)
        elapsed = time.time() - t0

        print(f"\n=== Gate A: {args.model} layer {L} ===")
        for k, v in res.items():
            print(f"  {k:18s}= {v:.4f}")
        print(f"  {'elapsed_s':18s}= {elapsed:.0f}")

        log_run({
            "script": "run_gate_a.py", "purpose": "gate_a_test",
            "model": args.model, "layer": L, "features": "dense", "sae_size": "",
            "dataset": "lebel", "subject": args.subject, "split": "test",
            "train_stories": args.train, "eval_stories": LB.TEST_STORY,
            "n_voxels": len(keep), "voxel_selection": "top single-rep r (test repeats)",
            "delays": args.delays, "alphas": "logspace(0,6,10)", "seed": SEED,
            "elapsed_s": round(elapsed, 1), "notes": args.notes, **res,
        })
    print("\nlogged to results/runs.csv")


if __name__ == "__main__":
    main()
