---
tags: [vocab, method]
aliases: [kernel ridge, kernel ridge regression, dual ridge]
---
# Kernel Ridge

**In one line:** the **same** [[Ridge Regression|ridge regression]], solved from a table of "how similar is scan *i* to scan *j*" instead of from the raw feature list.

**Why we care:** an [[Sparse Autoencoder (SAE)|SAE]] with 128K features × 4 [[Hemodynamic Delay|time lags]] is a 500K-column matrix, too big for a laptop. The similarity table is only (#scans × #scans), about 3,500 × 3,500. Same answer, much less memory.

**Our check:** on dense GPT-2 features, the normal and kernel versions give scores that match to 6 in 10 million. So dense and SAE scores come from the same model, which keeps the [[Encoding Model|pipeline]] fixed across configurations.

**Related:** [[Ridge Regression]] · [[himalaya]] · [[Encoding Model]]
