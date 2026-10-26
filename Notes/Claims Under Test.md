---
tags: [claims, core]
---
# Claims Under Test

The project exists to stress-test **four published claims**. Each gets scored in every one of the [[Experimental Grid|12 configurations]], with [[Controls and Baselines|controls]] attached.

## Claim 1 — Sparse SAE features ≈ dense hidden states
> Sparse [[Sparse Autoencoder (SAE)|SAE]] features predict brain activity about as well as the raw ([[Dense Hidden States|dense]]) model activations.
- **Source:** [[Lit - Sparse Features for Brain Responses (Lepori, Kay, Tuckute)]].
- **Why it matters:** if true, we lose nothing by moving to *interpretable* features — the whole research program rests on this.
- **How we test it:** [[Encoding Model|encoding score]] of SAE features vs dense features, same layer, same dataset, across the grid. Must beat [[Variance-Matched Random Features|random-feature]] and [[Untrained-Model Baseline|untrained]] baselines.
- **Failure mode to watch:** SAE features matching dense *only* at some sizes/models.

## Claim 2 — Semantic features do ~94% of the work
> [[Semantic Features|Semantic (meaning)]] features account for ~94% of the encoding performance; [[Syntactic Features|syntax]] and [[Lexical Features|word-form]] features add little.
- **Source:** [[Lit - Cortical Semantic Topography (Guo, Wu, Yiu)]].
- **Why it matters:** headline claim that "it's meaning" that makes models look brain-like.
- **How we test it:** score using only semantic features vs all features, per configuration. **Depends entirely on the [[Feature Taxonomy|taxonomy]]**, so we fix the category scheme first and report [[Inter-Rater Agreement (Kappa)|κ]].
- **Failure mode to watch:** the 94% is sensitive to how features are labeled; different labelers → different number.

## Claim 3 — Feature categories map onto distinct cortical regions
> Different *types* of meaning (objects, emotions, social, space, events) light up different [[Cortical Regions|cortical regions]], matching neuroscience priors.
- **Source:** [[Lit - Cortical Semantic Topography (Guo, Wu, Yiu)]].
- **Why it matters:** the most exciting and least-proven claim (small sample, small effects in the original).
- **How we test it:** per-category [[Voxel|voxelwise]] maps across the grid; check whether the same regions light up for the same categories across models/dictionaries/datasets.
- **Failure mode to watch:** category→region mapping that doesn't replicate across configurations.

## Claim 4 — Frontal language regions ≈ surprisal alone
> Frontal language regions are largely explained by [[Surprisal|surprisal]] (next-word predictability) on its own.
- **Source:** [[Lit - Sparse Features for Brain Responses (Lepori, Kay, Tuckute)]].
- **Why it matters:** tells us *which* brain areas are about meaning vs prediction.
- **How we test it:** compare a surprisal-only baseline against full feature models in frontal ROIs, per configuration.
- **Failure mode to watch:** surprisal explaining frontal regions in some datasets but not others.

## Scoring rule
A claim "survives" if it holds **across the grid** (not just in one cell) **above the control baselines** and **relative to the [[Noise Ceiling|noise ceiling]]**. For any failing claim, Phase 3 finds *which choice flips it* (model? size? dataset? layer?).
