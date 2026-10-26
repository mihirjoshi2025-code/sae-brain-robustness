"""Dense GPT-2 baseline on Pereira 2018 (exps 2, 3) at the frozen layers.

One feature per sentence (last token, passage context; see features/sentence_features),
no hemodynamic delays (one beta per sentence), himalaya ridge with 8 passage-grouped outer
folds and passage-grouped inner alpha CV, per-voxel corr of pooled out-of-fold
predictions. Ridge is solved as KernelRidgeCV on a linear kernel: identical to
RidgeCV (checked: max |dcorr| 2.8e-4 on P01 exp2 L7 language voxels) and faster since
n_sent < 768. Ceiling = across-subject ceiling with Brain-Score subject-pool extrapolation
(pipeline.pereira.ceilings); the k = n-1 pool ceiling is reported in notes.
Headline voxels: union of the participant's LH+RH language fROIs (fixed by the data,
independent of model and ceiling). Whole-brain numbers go in `notes`.
One results row per subject x experiment x layer; reruns skip logged cells.

Run: .venv/bin/python scripts/run_pereira_dense.py [--exps 2 3] [--subjects P01 ...]
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.sentence_features import sentence_feats  # noqa: E402
from pipeline import pereira as P  # noqa: E402
from pipeline.lebel import summarize  # noqa: E402
from pipeline.results import RESULTS_CSV, log_run  # noqa: E402

MODEL = "gpt2"
SCRIPT = "run_pereira_dense"


def done_cells() -> set:
    if not RESULTS_CSV.exists():
        return set()
    with open(RESULTS_CSV, newline="") as f:
        return {(r["subject"], r["eval_stories"], r["layer"]) for r in csv.DictReader(f)
                if r["script"] == SCRIPT}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--subjects", nargs="+", default=None)
    a = ap.parse_args()

    exp_of = {k: e for e, k in P.N_SENT.items()}
    for raw in sorted(glob.glob(f"{P.ROOT}/raw/*/data_*sentences.mat")):
        if not (Path(raw).parent / ".done").exists():
            continue  # download still streaming this subject
        subj, exp = Path(raw).parent.name, exp_of[int(Path(raw).name.split("_")[1][:3])]
        print(f"convert {subj} exp{exp}", flush=True)
        P.convert(subj, exp)

    cfg = yaml.safe_load(open(Path(__file__).resolve().parents[1] / "configs/layers.yaml"))
    layers = cfg["models"][MODEL]["layers"]  # {best, early, late}
    by_layer = {v: k for k, v in layers.items()}
    done = done_cells()
    bundle = None

    def bundle_fn():
        nonlocal bundle
        if bundle is None:
            from features.extract import load_model
            bundle = load_model(MODEL)
        return bundle

    for exp in a.exps:
        subs = [s for s in (a.subjects or P.SUBJECTS[exp]) if s in P.SUBJECTS[exp]
                and os.path.exists(P.h5_path(s, exp))]
        missing = [s for s in P.SUBJECTS[exp] if not os.path.exists(P.h5_path(s, exp))]
        if missing:  # the across-subject ceiling needs every participant
            print(f"exp{exp}: waiting for {missing}; skipping")
            continue
        ceil_all = P.ceilings(exp)
        ref = P.load(subs[0], exp)
        feats = sentence_feats(bundle_fn, MODEL, f"exp{exp}", ref.sentences,
                               ref.passage_ids, sorted(by_layer))
        for subj in subs:
            d = P.load(subj, exp)
            assert d.sentences == ref.sentences and np.array_equal(d.passage_ids, ref.passage_ids)
            t0 = time.time()
            c = ceil_all[subj]
            lang = c["mask"]
            assert np.array_equal(lang, P.language_mask(d))
            n_others = len(P.SUBJECTS[exp]) - 1
            for L in sorted(by_layer):
                if (subj, f"exp{exp}", str(L)) in done:
                    print(f"skip {subj} exp{exp} L{L}")
                    continue
                t1 = time.time()
                scores, best = P.fit_score([feats[L]], d.Y, d.passage_ids, kernel=True)
                s = summarize(scores[lang], c["extrap"])
                s_k = summarize(scores[lang], c["curve"][-1])
                wb = summarize(scores, c["pool_full"])
                med_alpha = float(np.median(np.stack(best)[:, lang]))
                row = dict(
                    script=SCRIPT, purpose="pereira_dense_baseline", model=MODEL, layer=L,
                    features="dense", dataset="pereira", subject=subj,
                    split=f"{P.N_OUTER}-fold passage-grouped CV (passage p -> fold p%{P.N_OUTER}); "
                          f"inner {P.N_INNER}-fold passage-grouped alpha CV",
                    train_stories=f"exp{exp} other folds", eval_stories=f"exp{exp}",
                    n_voxels=int(lang.sum()), voxel_selection="languageLH+RH fROIs (shipped atlas)",
                    delays="none", alphas=f"logspace(-1,9,21) per-voxel; median chosen {med_alpha:.3g}",
                    seed=0, elapsed_s=round(time.time() - t1, 1),
                    notes=(f"layer_role={by_layer[L]}; feature=last token w/ passage context + BOS; "
                           f"ceiling=across-subject, Brain-Score subject-pool extrapolation (k=1..{n_others}); "
                           f"k={n_others} pool ceiling lang={s_k['mean_ceiling']:.4f} "
                           f"normalized={s_k['normalized_mean']:.3f}; whole-brain (k={n_others} ceiling) V={len(scores)}: mean_corr={wb['mean_corr']:.4f} "
                           f"mean_ceiling={wb['mean_ceiling']:.4f} normalized={wb['normalized_mean']:.3f}"),
                    **s)
                log_run(row)
                print(f"{subj} exp{exp} L{L:>2} ({by_layer[L]}): lang corr {s['mean_corr']:.4f} "
                      f"ceil {s['mean_ceiling']:.4f} norm {s['normalized_mean']:.3f} | "
                      f"wb corr {wb['mean_corr']:.4f} ceil {wb['mean_ceiling']:.4f} "
                      f"[{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
