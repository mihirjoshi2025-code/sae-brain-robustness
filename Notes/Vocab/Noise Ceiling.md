---
tags: [vocab]
aliases: [noise ceiling, noise ceilings, explainable variance]
---
# Noise Ceiling

**In one line:** the **best score any model could possibly get** given how noisy the brain data is — the ceiling set by measurement noise, not by the model.

**Why we care:** a raw score is meaningless alone. A 0.10 is weak at a 0.40 ceiling but strong at a 0.12 ceiling. **Report the ceiling next to every score** — especially for small datasets like [[Pereira 2018|Pereira]].

**How it's estimated:** from repeated presentations / across-subject consistency. When the test target is an *average* of repeats, correct the single-repeat reliability with [[Spearman-Brown Correction|Spearman-Brown]] and take its square root to get a correlation-scale ceiling (this is what `pipeline/encoding.py::noise_ceiling_corr` does).

**Related:** [[Controls and Baselines]] · [[Encoding Model]] · [[fMRI]]
