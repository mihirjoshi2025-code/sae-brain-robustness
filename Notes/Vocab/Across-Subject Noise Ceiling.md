---
tags: [vocab]
aliases: [across-subject ceiling, inter-subject ceiling, ISC ceiling, subject-pool ceiling]
---
# Across-Subject Noise Ceiling

**In one line:** a [[Noise Ceiling|noise ceiling]] built from **other people's brains**: how well the responses of the other participants to the same stimuli predict this participant's [[Voxel|voxel]].

**When we use it:** when there are no repeated scans of the same person to compare, as in [[Pereira 2018]] (the public files hold one repetition-averaged image per sentence). With repeats we use [[Spearman-Brown Correction|Spearman-Brown]] instead, as for [[LeBel-Huth fMRI]].

**How (`pipeline/pereira.py::ceilings`):** fit the same [[Ridge Regression|ridge]] pipeline the model uses, but with the other participants' voxels as the features, with the same [[Passage-Grouped Cross-Validation|passage-held-out folds]]. The result is already a correlation, so normalized score = model corr / ceiling. Following the Brain-Score Pereira ceiling (Schrimpf et al. 2021), we repeat this with pools of 1, 2, … n−1 other people and fit a saturating curve to guess the ceiling with infinitely many people (**subject-pool extrapolation**).

**Caveats:** it only counts signal that is **shared across people**, so it underestimates the true ceiling. A model can beat it, and normalized scores come out on the generous side. The extrapolation is a curve fit to at most 7 points per voxel, so it's noisy; we also report the plain n−1 pool value.

**Related:** [[Noise Ceiling]] · [[Spearman-Brown Correction]] · [[Pereira 2018]]
