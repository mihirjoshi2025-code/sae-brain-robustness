---
tags: [team, planning]
---
# Roles and Workstreams

Four workstreams, one owner each. **Mihir = project lead** (integration, weekly syncs, final write-up).

> [!note] Critical path
> The **encoding pipeline** is the critical path. The other three workstreams build against **cached test data** until the pipeline is ready, then plug into the real thing.

## The four workstreams
### 1. Encoding pipeline (critical path)
- **Owns:** fMRI loading & alignment, [[Ridge Regression|ridge regression]] with [[Hemodynamic Delay|hemodynamic delays]], [[Noise Ceiling|noise ceilings]], scoring.
- **First deliverable:** dense-feature baseline reproduced on one dataset.
- **Tools:** [[himalaya]], Gallant lab voxelwise tutorials.

### 2. Feature extraction
- **Owns:** running stimuli through each model, caching [[Dense Hidden States|dense]] and [[Sparse Autoencoder (SAE)|SAE]] activations at matched layers.
- **First deliverable:** cached activations for one model, both datasets.
- **Tools:** [[SAELens]], 24GB-class GPU.

### 3. Feature taxonomy
- **Owns:** LLM-assisted labeling of SAE features, hand validation with [[Inter-Rater Agreement (Kappa)|inter-rater agreement]], matching features across dictionaries.
- **First deliverable:** labeled sample of 200 features with a κ score.
- **Key rule:** fix the [[Feature Taxonomy|category scheme]] before labeling (drives Claim 2).

### 4. Controls & statistics
- **Owns:** [[Contiguous Train-Test Split|contiguous splits]], [[Untrained-Model Baseline|untrained]] and [[Variance-Matched Random Features|random-feature]] baselines, [[Permutation Test|permutation tests]], results tables.
- **First deliverable:** control suite running on the dense baseline.

## Dependency order
```
Encoding pipeline ──┬─> Feature extraction plugs in
                    ├─> Controls & statistics plug in
                    └─> Feature taxonomy feeds category scores
```
Until the pipeline exists, everyone else develops against a small cached fixture so no one is blocked.

## Lead responsibilities (Mihir)
- Weekly syncs; keep the shared [[Results Table|results table]] honest.
- Own the config system and the `configs/` folder.
- Integration across the four folders (`pipeline/`, `features/`, `taxonomy/`, `controls/`).
- Final write-up and club presentation.
