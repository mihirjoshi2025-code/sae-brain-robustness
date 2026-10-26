---
tags: [vocab, stats]
aliases: [FDR, false discovery rate, Benjamini-Hochberg, BH correction]
---
# False Discovery Rate (FDR)

**In one line:** when you test thousands of voxels at once, FDR control keeps the **expected fraction of false positives among the voxels you call significant** at a chosen level (we use q = 0.05).

**Why we care:** at p < 0.05, 10,000 voxels with no real effect would still give ~500 "significant" ones by chance. The Benjamini-Hochberg procedure sorts the p-values and keeps only those under a rising threshold, so the reported "fraction of significant voxels" is honest. In `results/runs.csv` this is the `frac_voxels_sig` column (`controls/stats.py::fdr_bh`).

**Related:** [[Permutation Test]] · [[Block Permutation]] · [[Voxel]] · [[Results Table]]
