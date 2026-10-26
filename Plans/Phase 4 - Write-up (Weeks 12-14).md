---
tags: [plan, phase4]
---
# Phase 4 · Write-up · Weeks 12–14

**Goal:** turn the frozen results grid into a **short paper**, publish a **reproducible public repo**, and **present to the club**.

**Deliverables:** results grid + short paper · public repo with configs, cached results, and docs · club presentation.

> [!tip] The headline is a verdict: **which published claims survive the grid, and which choices flip the rest.**

---

## Week 12 — Draft the paper + build the results figure
- [ ] Build the **main results grid figure**: claims × 12 configs, pass/fail above baselines, with [[Noise Ceiling|noise ceilings]].
- [ ] Draft sections: question, grid, controls, per-claim findings, "which choice flips it," limitations.
- [ ] State clearly, per claim, **survives / partially survives / fails** and the conditions.
- [ ] Pull the numbers straight from the [[Results Table]] (no hand-copied figures).

---

## Week 13 — Reproducibility pass + internal review
- [ ] Clean the repo: `pipeline/ features/ taxonomy/ controls/ configs/` + a top-level README and a one-command "reproduce a cell" path.
- [ ] Verify a fresh run from a config reproduces a logged score (fixed [[Seed|seeds]]).
- [ ] Write docs: data download steps (no raw data in repo), cache layout, how to add a config.
- [ ] Internal review of the draft; incorporate feedback. Re-run arXiv scan — note any new papers to cite/address.

---

## Week 14 — Finalize + present
- [ ] Finalize the short paper and the results grid.
- [ ] Make the repo public (configs, cached results, docs; **not** raw fMRI).
- [ ] **Present findings to the club.**
- [ ] Write a short retrospective: what held up, what we'd change, open questions / next steps (incl. stretch goal if untouched).

---

## Final deliverables checklist
- [ ] Reproducible pipeline runnable from config files.
- [ ] Results grid across all 12 configurations with noise ceilings + baselines.
- [ ] Short write-up stating which of claims 1–4 survive and which choices flip them.
- [ ] Public repo (no raw data) + docs.
- [ ] Club presentation delivered.
