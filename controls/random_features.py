"""Variance-matched random features: same per-feature statistics, no content.

Definition used here: every feature column is permuted independently across all
words (pooled over the stories of the run), with a fixed seed. Each feature keeps
its exact value distribution -- mean, variance, and for sparse SAE codes its
sparsity (number of nonzero words) and nonzero values -- but loses its link to
which word it fired on. Width and scale are therefore identical to the real set,
so any score the real features have above this control comes from content, not
from dimensionality or scale. Signals that survive the shuffle (e.g. a feature's
mean times the number of words per TR, i.e. word rate) are correctly credited to
the control.

Works on dense arrays and scipy.sparse matrices (sparse stays sparse: only the
nonzero entries move, so cost scales with nnz, not N x width).

Not built here: NEXT_STEPS T6's random-projection "random dictionary"
(ReLU(dense @ R), top-k by the SAE's per-word L0). It needs real SAE codes.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def variance_matched(mats: list, seed: int = 0) -> list:
    """List of (N_i, d) feature matrices -> same shapes, columns shuffled."""
    rng = np.random.default_rng(seed)
    cuts = np.cumsum([m.shape[0] for m in mats])
    if sp.issparse(mats[0]):
        X = sp.vstack(mats).tocsc(copy=True)
        N = X.shape[0]
        for j in range(X.shape[1]):
            a, b = X.indptr[j], X.indptr[j + 1]
            if b > a:
                X.indices[a:b] = np.sort(rng.choice(N, b - a, replace=False))
                X.data[a:b] = rng.permutation(X.data[a:b])
        X = X.tocsr()
    else:
        X = rng.permuted(np.concatenate([np.asarray(m) for m in mats]), axis=0)
    return [X[s:e] for s, e in zip(np.r_[0, cuts[:-1]], cuts)]


def provider(base, stories: list, seed: int = 0):
    """Wrap a feature provider (story -> (N, d)) into its random-matched control."""
    out = dict(zip(stories, variance_matched([base(st) for st in stories], seed)))
    return lambda story: out[story]
