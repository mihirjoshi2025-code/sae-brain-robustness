---
tags: [plan, phase1]
---
# Phase 1 · Setup · Weeks 1–3

**Goal:** stand up the infrastructure and **reproduce the dense-feature baseline on one dataset**. Everything downstream plugs into this.

**Gate A (end of Week 3):** dense baseline reproduced. **Miss → cut to one dataset.**

> [!tip] Strategy
> Get a *thin vertical slice* working end-to-end on **one model, one dataset, dense features** before scaling to the grid. Correctness first, breadth later.

---

## Week 1 — Infrastructure & onboarding
**Everyone**
- [ ] Finish the three kickoff papers ([[_Reading List]]).
- [ ] Create/clone the repo; agree on folder layout (`pipeline/ features/ taxonomy/ controls/ configs/`) and a `README`.
- [ ] Confirm **compute source** and get access (GPU for extraction, CPU for ridge).
- [ ] Set up shared storage for raw data + cached activations (outside the repo). Add `.gitignore` for data/weights.

**Encoding pipeline (critical path)**
- [ ] Download **one** fMRI dataset from OpenNeuro to shared storage (likely [[LeBel-Huth fMRI]]). **Never commit raw data.**
- [ ] Work through a Gallant-lab voxelwise tutorial to learn the [[himalaya]] / [[Ridge Regression|ridge]] workflow.

**Feature extraction**
- [ ] Stand up model loading for the first candidate model; confirm GPU runs.
- [ ] Define the **cache file naming** scheme: one file per `model × layer × dataset × SAE size`.

**Feature taxonomy**
- [ ] Draft the **category scheme** (semantic / syntactic / lexical / prediction / other). **Freeze it before any labeling.**

**Controls & statistics**
- [ ] Build the small **cached test fixture** so the team isn't blocked on the pipeline.
- [ ] Draft the [[Results Table]] schema (config id, seed, baselines, scores, noise ceiling, p-value).

---

## Week 2 — First dense baseline (one model, one dataset)
**Encoding pipeline**
- [ ] Implement fMRI loading + alignment of stimulus timing to TRs.
- [ ] Add [[Hemodynamic Delay|hemodynamic-delay]] (lagged FIR) predictors.
- [ ] Fit [[Ridge Regression|ridge]] [[Encoding Model|encoding]] from **[[Dense Hidden States|dense]]** features of one middle [[Layer|layer]]; score on a held-out **[[Contiguous Train-Test Split|contiguous split]]**.

**Feature extraction**
- [ ] Extract + cache **dense** activations for the first model on the dataset (early/middle/late layers).

**Controls & statistics**
- [ ] Implement [[Contiguous Train-Test Split|contiguous splits]] + [[Noise Ceiling|noise-ceiling]] estimation.
- [ ] Wire the control suite to run on the dense baseline (even if some baselines are stubs).

**Feature taxonomy**
- [ ] Pilot [[LM-Assisted Labeling|LLM labeling]] on ~20 features; sanity-check the prompt + output format.

---

## Week 3 — Reproduce & lock Gate A; pick layers
**Encoding pipeline**
- [ ] Reproduce the **published dense-feature baseline** numbers (within tolerance) on the one dataset. ← **Gate A**
- [ ] **Pick the layers** (best + one early + one late) per the first model **before looking at any SAE results**. Record them.

**Feature extraction**
- [ ] Confirm cache format works; begin dense extraction for a second dataset if ahead.

**Feature taxonomy**
- [ ] Finalize + **freeze the category scheme**; write labeling guidelines for raters.

**Controls & statistics**
- [ ] Control suite produces sensible numbers on the dense baseline (word rate, sentence position, surprisal, random features, untrained model). Log everything to the [[Results Table]].

**Lead**
- [ ] **Gate A review.** If dense baseline not reproduced → **cut to one dataset** and adjust the roadmap. Record decision.
- [ ] Finalize the **model list** now that SAE coverage is known.

---

## Exit criteria (all true to pass Gate A)
- [ ] Dense baseline reproduced on one dataset, logged to the results table.
- [ ] Layers chosen and frozen (no peeking at SAE results).
- [ ] Category scheme frozen; labeling guidelines written.
- [ ] Control suite + noise ceilings running on the dense baseline.
- [ ] Final model list decided from SAE availability.
