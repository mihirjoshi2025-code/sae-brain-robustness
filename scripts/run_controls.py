"""Control suite for one configuration (model x feature set x layer) on LeBel/Huth.

Same setup as run_gate_a.py (train on the 10 training stories, test on the
10-repeat mean of `wheretheressmoke`, top-10k voxels by single-repeat r,
Spearman-Brown ceiling). Logs one row per feature set to `results/runs.csv`:
  - the main feature set (control blank), with its permutation test
  - untrained       same model, random init, dense, same layer
  - random_matched  main features with each column shuffled across words
  - word_rate, sentence_position, surprisal   (layer-independent, run once)
Every row carries p_value (mean corr), n_perm and frac_voxels_sig (BH-FDR 0.05)
from a block permutation test on the test predictions (no refitting).

Feature sets are providers: `factory(model, layer, layers) -> (story -> (N_words, d))`.
`--features dense` is built in; any other value is `package.module:factory`,
imported at run time, so new feature sets (e.g. SAE codes) plug in without edits.

Usage:
  python scripts/run_controls.py --frozen
  python scripts/run_controls.py --features dense --layer 7
  python scripts/run_controls.py --features features.sae:provider --layer 7
"""
import argparse
import importlib
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from controls import baselines as B  # noqa: E402
from controls import random_features as RF  # noqa: E402
from controls import stats as S  # noqa: E402
from controls import untrained as U  # noqa: E402
from features.cache import word_feats  # noqa: E402
from features.extract import load_model  # noqa: E402
from pipeline import data as D  # noqa: E402
from pipeline import lebel as LB  # noqa: E402
from pipeline.encoding import noise_ceiling_corr  # noqa: E402
from pipeline.results import log_run  # noqa: E402

BASELINES = ["word_rate", "sentence_position", "surprisal"]
CONTROLS = ["untrained", "random_matched"] + BASELINES


def dense_provider(model: str, layer: int, layers=None):
    layers = sorted(set(layers or []) | {layer})
    _b = []

    def bundle():
        if not _b:
            print(f"loading {model}...")
            _b.append(load_model(model))
        return _b[0]

    return lambda story: word_feats(bundle, model, story, layers)[layer]


PROVIDERS = {"dense": dense_provider}


def resolve(name: str):
    if name in PROVIDERS:
        return PROVIDERS[name]
    mod, fn = name.split(":")
    return getattr(importlib.import_module(mod), fn)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--layer", type=int, nargs="+", default=[7])
    g.add_argument("--frozen", action="store_true",
                   help="run the layers frozen in configs/layers.yaml")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--features", default="dense",
                    help="'dense' or package.module:factory")
    ap.add_argument("--controls", nargs="+", default=CONTROLS, choices=CONTROLS)
    ap.add_argument("--no-main", action="store_true", help="skip the main feature row")
    ap.add_argument("--train", nargs="+", default=LB.TRAIN_STORIES)
    ap.add_argument("--subject", default="UTS01")
    ap.add_argument("--voxels", type=int, default=10000)
    ap.add_argument("--delays", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--block", type=int, default=10, help="permutation block (TRs)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--notes", default="")
    args = ap.parse_args()
    np.random.seed(args.seed)

    if args.frozen:
        cfg = yaml.safe_load(open(ROOT / "configs" / "layers.yaml"))
        layers = sorted(set(cfg["models"][args.model]["layers"].values()))
    else:
        layers = args.layer

    reps = D.load_repeats(LB.TEST_STORY, args.subject)
    nc = noise_ceiling_corr(reps)
    keep = np.argsort(nc["single_rep_r"])[::-1][:args.voxels]
    ceil = nc["ceiling"][keep]
    Ytr = [D.load_bold(st, args.subject)[:, keep] for st in args.train]
    Yte = reps.mean(0)[:, keep]
    del reps
    stories = args.train + [LB.TEST_STORY]
    print(f"kept {len(keep)} voxels; ceiling mean={ceil.mean():.3f}; "
          f"test BOLD autocorr < 0.1 after {S.autocorr_length(Yte)} TRs; block={args.block}")

    base_row = {
        "script": "run_controls.py", "purpose": "controls", "model": args.model,
        "dataset": "lebel", "subject": args.subject, "split": "test",
        "train_stories": args.train, "eval_stories": LB.TEST_STORY,
        "n_voxels": len(keep), "voxel_selection": "top single-rep r (test repeats)",
        "delays": args.delays, "alphas": "logspace(0,6,10)", "seed": args.seed,
        "n_perm": args.n_perm,
    }
    table = []

    def run(provider, row):
        t0 = time.time()
        wf = {st: provider(st) for st in stories}
        wf = {st: (x.toarray() if sp.issparse(x) else np.asarray(x)) for st, x in wf.items()}
        Xtr = [LB.tr_features(wf[st], st, len(y)) for st, y in zip(args.train, Ytr)]
        Xte = LB.tr_features(wf[LB.TEST_STORY], LB.TEST_STORY, len(Yte))
        scores, pred, yz = LB.fit_score(Xtr, Ytr, Xte, Yte, delays=args.delays,
                                        return_pred=True)
        res = LB.summarize(scores, ceil)
        perm = S.block_permutation(yz, pred, args.block, args.n_perm, args.seed)
        res["p_value"] = perm["p_mean"]
        res["frac_voxels_sig"] = float(S.fdr_bh(perm["p_voxel"]).mean())
        elapsed = time.time() - t0
        notes = f"block={args.block}TR, BH-FDR q=0.05, d={wf[LB.TEST_STORY].shape[1]}"
        log_run({**base_row, **row, **res, "elapsed_s": round(elapsed, 1),
                 "notes": "; ".join(filter(None, [notes, args.notes]))})
        table.append((row.get("layer", ""), row["control"] or row["features"], res))
        print(f"  L{row.get('layer') if row.get('layer') != '' else '-'} {row['control'] or row['features']:18s} "
              f"r={res['mean_corr']:.4f} norm={res['normalized_mean']:.3f} "
              f"p={res['p_value']:.4f} sig={res['frac_voxels_sig']:.3f} ({elapsed:.0f}s)")

    factory = resolve(args.features)
    for L in layers:
        main_prov = factory(args.model, L, layers)
        lrow = {"layer": L, "sae_size": ""}
        if not args.no_main:
            run(main_prov, {**lrow, "features": args.features, "control": ""})
        if "untrained" in args.controls:
            run(U.provider(args.model, L, layers, seed=args.seed),
                {**lrow, "features": "dense", "control": "untrained"})
        if "random_matched" in args.controls:
            run(RF.provider(main_prov, stories, seed=args.seed),
                {**lrow, "features": args.features, "control": "random_matched"})

    fns = {"word_rate": B.word_rate, "sentence_position": B.sentence_position,
           "surprisal": lambda st: B.surprisal(st, args.model)}
    for name in [c for c in args.controls if c in BASELINES]:
        run(fns[name], {"layer": "", "features": name, "control": name,
                        "model": args.model if name == "surprisal" else ""})

    print("\nlayer  feature set          mean_r   norm    p      frac_sig")
    for L, name, r in table:
        print(f"{str(L):6s} {name:20s} {r['mean_corr']:.4f}  {r['normalized_mean']:.3f}  "
              f"{r['p_value']:.4f} {r['frac_voxels_sig']:.3f}")
    print("logged to results/runs.csv")


if __name__ == "__main__":
    main()
