---
tags: [plan, phase2]
---
# Phase 2 · Core grid · Weeks 4–8

**Goal:** run **all 12 configurations** (3 models × 2 SAE sizes × 2 datasets), label SAE features and validate a sample by hand, and apply the **full control suite** to every configuration.

**Gate B (end of Week 8):** controls run on **all 12 configs**; taxonomy agreement **κ ≥ 0.7.**

> [!tip] Strategy
> Extraction is the expensive step — **cache aggressively** and never recompute. Labeling and control code should be config-agnostic so adding a cell is cheap.

---

## Week 4 — Scale extraction + add SAE features
**Feature extraction**
- [ ] Load public [[Sparse Autoencoder (SAE)|SAEs]] via [[SAELens]] ([[Gemma Scope]] / [[Llama Scope]] / GPT-2 SAEs) at the frozen [[Layer|layers]].
- [ ] Extract + cache **[[Dense Hidden States|dense]] and SAE** activations for **model 1, both datasets, both [[Dictionary Size|sizes]]** → 4 cells.
**Encoding pipeline**
- [ ] Generalize the pipeline to read any config file and run SAE-feature encoding.
**Controls & statistics**
- [ ] Make the control suite fully config-driven (runs per cell automatically).

---

## Week 5 — Finish extraction for all 3 models
**Feature extraction**
- [ ] Cache activations for **models 2 and 3**, both datasets, both sizes → remaining 8 cells. All 12 cells cached.
- [ ] Verify cache integrity (shapes, no NaNs, correct layer/size tags).
**Encoding pipeline**
- [ ] Smoke-test the encoding run on 2–3 cells end-to-end.

---

## Week 6 — Run the grid; start labeling
**Encoding pipeline**
- [ ] Run SAE + dense [[Encoding Model|encoding]] across **all 12 configs**; write scores + [[Noise Ceiling|noise ceilings]] to the [[Results Table]].
**Feature taxonomy**
- [ ] Run [[LM-Assisted Labeling|LLM labeling]] over features for each dictionary.
- [ ] Begin **matching features across dictionaries** (16K ↔ 65K) per model.
**Controls & statistics**
- [ ] Attach the full control suite to every cell: [[Untrained-Model Baseline|untrained]], [[Variance-Matched Random Features|random features]], [[Word Rate|word rate]], [[Sentence Position|sentence position]], [[Surprisal|surprisal]].

---

## Week 7 — Hand validation + inter-rater agreement
**Feature taxonomy**
- [ ] Two raters independently label a **200-feature sample**; compute [[Inter-Rater Agreement (Kappa)|κ]].
- [ ] If **κ < 0.7**: refine guidelines (not the categories — scheme is frozen), re-label, recompute.
**Controls & statistics**
- [ ] Add [[Permutation Test|permutation-test]] significance per cell; confirm [[Contiguous Train-Test Split|splits]] applied everywhere.
**Encoding pipeline**
- [ ] Re-run any cells that changed; keep the results table authoritative.

---

## Week 8 — Fill the grid; Gate B
**All**
- [ ] Every cell has: dense score, SAE score, all baseline scores, noise ceiling, p-value — logged.
- [ ] Spot-check a few cells for plausibility (middle layer > early/late; SAE ≈ dense roughly).
**Lead**
- [ ] **Gate B review:** controls on all 12 configs ✔ and κ ≥ 0.7 ✔. If missed → **narrow scope** (drop a model or a size) and record the decision.

---

## Exit criteria (all true to pass Gate B)
- [ ] All 12 cells extracted, cached, and scored with baselines + noise ceilings.
- [ ] Taxonomy labels done + cross-dictionary matching; validated sample **κ ≥ 0.7**.
- [ ] Permutation-test significance attached to every cell.
- [ ] Results table complete enough to start claim scoring in Phase 3.
