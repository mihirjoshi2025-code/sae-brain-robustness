---
tags: [vocab]
aliases: [Spearman-Brown, Spearman–Brown, Spearman-Brown prophecy formula]
---
# Spearman-Brown Correction

**In one line:** a formula that says **how much more reliable an average gets** when you average several noisy repeats of the same measurement.

**The formula:** if one repeat has reliability *r*, the average of *N* repeats has reliability **N·r / (1 + (N−1)·r)**.

**Why we care:** our test story was heard **10 times**, and we predict the *average* of those 10 scans. Averaging cancels a lot of noise, so the average is far more predictable than any single scan. Using the single-scan reliability as the [[Noise Ceiling|ceiling]] made normalized scores come out impossibly high (~2.9). After the correction, the ceiling on the correlation scale is **√(corrected reliability)**, and the Gate A normalized score is a sensible ~0.35.

**Example (UTS01, top 10k voxels):** single-scan *r* ≈ 0.076 → 10-repeat reliability ≈ 0.42 → correlation ceiling ≈ 0.65.

**Related:** [[Noise Ceiling]] · [[LeBel-Huth fMRI]] · [[Encoding Model]]
