---
tags: [vocab, stats, control]
aliases: [block permutation, block permutation test, block shuffle]
---
# Block Permutation

**In one line:** a [[Permutation Test]] for time series: cut the model's test-set predictions into **contiguous blocks** (we use 10 TRs = 20 s), shuffle the block order, and re-score against the real brain data to see what correlation chance alone gives.

**Why blocks, not single time points:** fMRI signals are smooth in time (neighboring TRs are correlated). Shuffling single TRs destroys that smoothness and makes the chance distribution too narrow, so almost everything looks significant. Blocks longer than the autocorrelation (~5 TRs here) keep the smoothness inside each block.

**How we use it:** the model is **never refit** -- only its predictions on `wheretheressmoke` are shuffled, 1,000 times (seed 0). This gives a p-value per [[Voxel]], a p-value for the mean correlation, and, after [[False Discovery Rate (FDR)|FDR]] correction, the fraction of significant voxels (`controls/stats.py::block_permutation`).

**Related:** [[Permutation Test]] · [[Contiguous Train-Test Split]] · [[Controls and Baselines]]
