---
tags: [plan, phase3]
---
# Phase 3 · Analysis · Weeks 9–11

**Goal:** score **claims 1–4** in each configuration, and for any failing claim, **find which choice flips it** (model? [[Dictionary Size|size]]? dataset? [[Layer|layer]]?). Optional checkpoint stretch goal if ahead.

**Gate C (end of Week 11):** **every claim scored**; noise ceilings reported for each.

> [!tip] The analysis question isn't "did it work?" but "**under which choices does it hold, and against which baselines?**"

---

## Week 9 — Score the four claims across the grid
See [[Claims Under Test]] for definitions.
- [ ] **Claim 1** (SAE ≈ dense): per cell, compare SAE vs [[Dense Hidden States|dense]] [[Encoding Model|encoding]] scores, above [[Variance-Matched Random Features|random]] + [[Untrained-Model Baseline|untrained]] baselines.
- [ ] **Claim 2** (semantic ~94%): per cell, score **[[Semantic Features|semantic]]-only** vs **all** features; report the fraction. Note sensitivity to the [[Feature Taxonomy|taxonomy]].
- [ ] **Claim 3** (category→region): per cell, build per-category [[Voxel|voxelwise]]/[[Cortical Regions|ROI]] maps; check category→region consistency.
- [ ] **Claim 4** (frontal ≈ surprisal): per cell, compare [[Surprisal|surprisal]]-only vs full models in frontal ROIs.
- [ ] Record each claim's result **relative to the [[Noise Ceiling|noise ceiling]]** and baselines.

---

## Week 10 — Robustness dissection
- [ ] Build the **survival grid**: for each claim, mark pass/fail in each of the 12 cells.
- [ ] For each **failing** claim, run the "**which choice flips it**" analysis — vary one axis at a time (model / size / dataset / layer) holding others fixed.
- [ ] Check **cross-dataset agreement**: does a claim hold in both [[LeBel-Huth fMRI|naturalistic]] and [[Pereira 2018|sentence]] data? (Watch small-data noise.)
- [ ] Sensitivity check for **Claim 2** on taxonomy: how much does the 94% move if labels shift within [[Inter-Rater Agreement (Kappa)|κ]] uncertainty?

---

## Week 11 — Finalize scoring; Gate C; (optional) stretch goal
- [ ] Confirm **every claim is scored in every cell** with noise ceilings + significance. ← **Gate C**
- [ ] Freeze the results grid for the write-up.
- [ ] **Optional stretch goal (only if ahead):** checkpoint analysis across training using [[Pythia]] checkpoints — see [[Lit - From Language to Cognition (AlKhamissi)]] and [[Lit - Training-Driven Representational Geometry]].
**Lead**
- [ ] **Gate C review.** If behind → **drop the stretch goal**, lock scope, move to write-up. Record decision.

---

## Exit criteria (all true to pass Gate C)
- [ ] All four claims scored in all 12 cells, with baselines + noise ceilings.
- [ ] Survival grid complete; flip-analysis done for each failing claim.
- [ ] Cross-dataset and taxonomy-sensitivity checks recorded.
- [ ] Results grid frozen for Phase 4.
