---
tags: [vocab]
aliases: [encoding model, encoding, brain encoding, brain score]
---
# Encoding Model

**In one line:** a simple formula that predicts **brain activity from model features** — fit features → [[Voxel|voxel]] activity on training data, then measure prediction accuracy on held-out data ("the brain score").

**In this project:** we build encoding models from [[Dense Hidden States|dense]] and [[Sparse Autoencoder (SAE)|SAE]] features, via [[Ridge Regression|ridge regression]] with [[Hemodynamic Delay|hemodynamic delays]], and compare scores across the grid.

**The warning:** encoding scores inflate easily — always judge against [[Controls and Baselines|baselines]] and the [[Noise Ceiling|noise ceiling]].

**Related:** [[Ridge Regression]] · [[Noise Ceiling]] · [[Voxel]]
