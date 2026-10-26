---
tags: [vocab, control]
aliases: [contiguous split, contiguous train/test split, blocked split]
---
# Contiguous Train-Test Split

**In one line:** split the data into **continuous blocks** of time (e.g., first 80% train, last 20% test) instead of randomly shuffling time points.

**Why it's a control:** neighboring [[fMRI]] time points (TRs) are **autocorrelated** — random shuffling leaks test info into training and **inflates** the score. Contiguous splits prevent that leakage.

**Related:** [[Controls and Baselines]] · [[fMRI]] · [[Permutation Test]]
