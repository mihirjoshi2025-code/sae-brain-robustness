# Brain–LLM SAE Robustness

Source plan: `Downloads/Brain–LLM SAE Robustness Execution Plan(1).pdf` (Oct 14, 2026, Mihir Joshi).
Full breakdown: `Notes/`. Week-by-week plan: `Plans/`.

## Research question (PROTECTED)
Do SAE-based brain-encoding results reflect how the brain actually processes language, or are they artifacts of **model**, **dictionary**, and **evaluation** choices? We re-run recent claims across a grid of configurations under strict confound controls and report which claims survive.

## The experimental grid (PROTECTED)
**3 models × 2 SAE sizes × 2 datasets = 12 configurations.** The encoding pipeline is held **fixed** across all 12.

| Axis | Options |
|---|---|
| Model | 3 open LLMs with public SAEs (candidates: GPT-2 family, Gemma 2 + Gemma Scope, Llama 3.1 + Llama Scope). Final pick depends on SAE layer coverage. |
| SAE dictionary | 2 sizes per model (e.g., 16K and 65K features). |
| fMRI dataset | Naturalistic stories (LeBel/Huth podcast fMRI) + isolated sentences (Pereira 2018). |
| Layer | Best-predicting layer per model, plus one early and one late. **Pick layers before looking at SAE results.** |

## The four claims under test (PROTECTED)
1. Sparse SAE features predict brain activity about as well as dense hidden states.
2. Semantic features account for most of the encoding performance (reported ~94%).
3. Feature categories map onto distinct cortical regions.
4. Frontal language regions are largely explained by surprisal alone.

## Controls applied to EVERY configuration (PROTECTED)
- Contiguous train/test splits (never random-shuffle TRs).
- Untrained-model baseline.
- Variance-matched random SAE features.
- Simple baselines: word rate, sentence position, surprisal.
- Report **noise ceilings** alongside every score.
- Permutation tests for significance.

## Rules / methodology guardrails (PROTECTED)
- **Fix layers before looking at SAE results.** No peeking.
- **Fix the feature category scheme before labeling.** Report inter-rater agreement (κ).
- **Never commit raw fMRI data to the repo.** Download once to a shared location.
- **Cache activations** as one file per `model × layer × dataset × SAE size`. Never recompute.
- **Log every run's config and score** to a shared results table. Fix random seeds.
- A **missed gate means narrowing scope, not stopping** (e.g., cut to one dataset).

## Phase gates (PROTECTED)
- **Gate A** (end Phase 1): dense baseline reproduced. Miss → cut to one dataset.
- **Gate B** (end Phase 2): controls run on all 12 configs; taxonomy agreement κ ≥ 0.7.
- **Gate C** (end Phase 3): every claim scored; noise ceilings reported for each.

## Roles / workstreams
Four workstreams, one owner each. Encoding pipeline is the **critical path**; the other three build against cached test data until it's ready. Mihir = project lead (integration, weekly syncs, final write-up).

| Workstream | Owns | First deliverable |
|---|---|---|
| Encoding pipeline | fMRI loading/alignment, ridge regression with hemodynamic delays, noise ceilings, scoring | Dense-feature baseline reproduced on one dataset |
| Feature extraction | Running stimuli through each model, caching dense + SAE activations at matched layers | Cached activations for one model, both datasets |
| Feature taxonomy | LLM-assisted labeling of SAE features, hand validation + inter-rater agreement, matching features across dictionaries | Labeled sample of 200 features with a κ score |
| Controls & statistics | Contiguous splits, untrained/random baselines, permutation tests, results tables | Control suite running on the dense baseline |

## Tooling (suggested)
- `himalaya` — ridge regression (voxelwise encoding).
- `SAELens` — loading public SAEs (Gemma Scope, Llama Scope, etc.).
- Gallant lab voxelwise tutorials — onboarding.
- Compute: one 24GB-class GPU for activation extraction; ridge regression on CPU. **Open: club cluster / dept resources / cloud credits.**

## Repo layout (planned)
```
pipeline/    features/    taxonomy/    controls/    configs/   # one config file per 12 configs
```

## Key risks / open decisions
- Pipeline stalls → if dense baseline not reproduced by end of Phase 1, cut to one dataset.
- SAE coverage → public SAEs may not exist at the right layers/sizes; **decide final model list after checking availability.**
- Taxonomy subjectivity → category labels drive claim 2; fix scheme first, report κ.
- Small-dataset noise → sentence-level data has few stimuli; report noise ceilings.
- Getting scooped → check arXiv weekly; adjust claims if a new paper covers them.
- Open: final model list, compute source, whether to attempt the training-checkpoint stretch goal.

## Conventions for working in this folder
- This is an Obsidian-style vault: notes use `[[wikilinks]]`. Vocab lives in `Notes/Vocab/`, literature summaries in `Notes/Literature/`.
- Plain-English first. Every piece of jargon should have a `Vocab/` entry.
- Keep large artifacts (raw fMRI, cached activations, model weights) **outside** this vault.
- The folder name has a dash, not spaces, but still quote paths in shell when in doubt.
