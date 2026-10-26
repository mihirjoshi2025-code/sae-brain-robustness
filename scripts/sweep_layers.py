"""Layer sweep + freeze (no-peeking rule): pick layers BEFORE any SAE result.

Selection never touches the test story. Each layer is scored by story-level
cross-validation inside the training stories (5 folds x 2 held-out stories);
per-fold scores go to `results/runs.csv` with split=val.

Pre-registered rule (fixed before running):
  best  = layer with the highest mean CV correlation (top-N reliable voxels)
  early = round(0.25 * (n_layers - 1))
  late  = n_layers - 1
The choice is written to `configs/layers.yaml`; an existing entry for the model is
never overwritten (that would be re-picking after the fact).

Usage:
  python scripts/sweep_layers.py --model gpt2
"""
import argparse
import sys
import time
from datetime import date
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
N_FOLDS = 5
CONFIG = ROOT / "configs" / "layers.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--subject", default="UTS01")
    ap.add_argument("--voxels", type=int, default=10000)
    ap.add_argument("--delays", type=int, nargs="+", default=[1, 2, 3, 4])
    args = ap.parse_args()
    np.random.seed(SEED)

    cfg = yaml.safe_load(open(CONFIG)) if CONFIG.exists() else {}
    cfg = cfg or {}
    if args.model in cfg.get("models", {}):
        sys.exit(f"{args.model} layers already frozen in {CONFIG}; refusing to re-pick.")

    from transformers import AutoConfig
    n_layers = AutoConfig.from_pretrained(args.model).num_hidden_layers
    layers = list(range(n_layers))

    _bundle = []
    def bundle():
        if not _bundle:
            print(f"loading {args.model}...")
            _bundle.append(load_model(args.model))
        return _bundle[0]

    # Voxel subset: model-agnostic reliability from the test repeats (same as Gate A).
    reps = D.load_repeats(LB.TEST_STORY, args.subject)
    keep = np.argsort(noise_ceiling_corr(reps)["single_rep_r"])[::-1][:args.voxels]
    del reps

    stories = LB.TRAIN_STORIES
    Y = {st: D.load_bold(st, args.subject)[:, keep] for st in stories}
    print("caching word activations for all layers...")
    wf = {st: word_feats(bundle, args.model, st, layers) for st in stories}
    folds = np.array_split(np.array(stories), N_FOLDS)

    cv = {}
    for L in layers:
        X = {st: LB.tr_features(wf[st][L], st, len(Y[st])) for st in stories}
        fold_means = []
        for k, val in enumerate(folds):
            t0 = time.time()
            tr = [s for s in stories if s not in val]
            scores = LB.fit_score([X[s] for s in tr], [Y[s] for s in tr],
                                  [X[s] for s in val], [Y[s] for s in val],
                                  delays=args.delays)
            res = LB.summarize(scores)
            fold_means.append(res["mean_corr"])
            log_run({
                "script": "sweep_layers.py", "purpose": "layer_selection",
                "model": args.model, "layer": L, "features": "dense", "sae_size": "",
                "dataset": "lebel", "subject": args.subject, "split": f"val_fold{k}",
                "train_stories": tr, "eval_stories": list(val),
                "n_voxels": len(keep), "voxel_selection": "top single-rep r (test repeats)",
                "delays": args.delays, "alphas": "logspace(0,6,10)", "seed": SEED,
                "elapsed_s": round(time.time() - t0, 1), **res,
            })
        cv[L] = float(np.mean(fold_means))
        print(f"layer {L:2d}: CV mean corr = {cv[L]:.4f}  (folds {np.round(fold_means, 3)})")

    best = max(cv, key=cv.get)
    early = int(round(0.25 * (n_layers - 1)))
    late = n_layers - 1
    cfg.setdefault("models", {})[args.model] = {
        "n_layers": n_layers,
        "layers": {"best": best, "early": early, "late": late},
        "frozen_on": date.today().isoformat(),
        "selection": (f"best = max mean story-level {N_FOLDS}-fold CV corr within the "
                      f"{len(stories)} training stories (subject {args.subject}, top "
                      f"{args.voxels} voxels); early = round(0.25*(n_layers-1)); "
                      f"late = n_layers-1. Test story never used. Chosen before any SAE result."),
        "cv_mean_corr": {int(k): round(v, 4) for k, v in cv.items()},
    }
    CONFIG.parent.mkdir(exist_ok=True)
    with open(CONFIG, "w") as f:
        f.write("# FROZEN layer choices (no-peeking rule). Written by scripts/sweep_layers.py.\n"
                "# Do not edit by hand; never re-pick after seeing SAE results.\n")
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"\nFROZEN {args.model}: best={best} early={early} late={late} -> {CONFIG}")


if __name__ == "__main__":
    main()
