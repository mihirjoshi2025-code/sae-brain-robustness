---
tags: [controls, core]
---
# Controls and Baselines

The point of this project is **not** to produce another high brain score — it's to find out which scores are **real**. That's what the controls are for. Every configuration runs the full suite.

> [!warning] Why we need them
> A random projection of model activations can predict the brain surprisingly well, and small datasets inflate apparent alignment. See [[Lit - What Are LLMs Mapping To (Feghhi, Hadidi)]] and [[Lit - Illusions of Alignment]]. Without baselines, a "94%" or a "region map" can be noise.

## The control suite (applied to every configuration)
| Control | What it checks | Why |
|---|---|---|
| [[Contiguous Train-Test Split|Contiguous train/test splits]] | Don't leak information across neighboring time points | fMRI TRs are autocorrelated; random shuffling inflates scores |
| [[Untrained-Model Baseline|Untrained-model baseline]] | How much score comes from architecture alone, not learning | A randomly-initialized model shouldn't "know" meaning |
| [[Variance-Matched Random Features|Variance-matched random SAE features]] | Whether SAE features beat random features of the same scale | Controls for dimensionality/variance, not content |
| [[Word Rate]] baseline | Low-level timing confound | Brain responds to *how many words per second* |
| [[Sentence Position]] baseline | Narrative/position confound | Brain responds to where you are in a passage |
| [[Surprisal]] baseline | Predictability confound (also = Claim 4) | Brain tracks next-word surprise independent of meaning |

## Noise ceilings (report alongside every score)
A [[Noise Ceiling|noise ceiling]] is the **best score any model could achieve** given how noisy the brain data is. Report it next to every number. A score of 0.10 means something very different at a 0.12 ceiling vs a 0.40 ceiling. Essential for the sentence-level [[Pereira 2018|Pereira]] data, which has few stimuli.

## Statistics
- [[Permutation Test|Permutation tests]] for significance (not parametric assumptions).
- Fixed random [[Seed|seeds]], logged per run.
- Every run's config + score written to a shared [[Results Table|results table]].

## No-peeking rules
- **Pick layers before looking at SAE results.**
- **Fix the [[Feature Taxonomy|category scheme]] before labeling features**, and report [[Inter-Rater Agreement (Kappa)|κ]].
- These are the two places the project is most vulnerable to fooling itself.
