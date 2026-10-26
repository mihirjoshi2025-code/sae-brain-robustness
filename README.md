# SAE–Brain Robustness

Do SAE-based brain-encoding results reflect how the brain processes language, or are they artifacts of **model**, **dictionary**, and **evaluation** choices? We re-run recent brain↔LLM claims across a grid of configurations under strict confound controls and report which survive.

## The study
- **Grid:** 3 models × 2 SAE sizes × 2 fMRI datasets = **12 configurations**, one fixed encoding pipeline.
- **Claims under test:** (1) sparse SAE features ≈ dense hidden states; (2) semantic features do ~94% of encoding; (3) feature categories map to cortical regions; (4) frontal regions ≈ surprisal alone.
- **Controls (every config):** contiguous splits, untrained-model baseline, variance-matched random features, word-rate / sentence-position / surprisal baselines, noise ceilings, permutation tests.

## Repo layout
| Path | What |
|---|---|
| `pipeline/` | Voxelwise ridge encoding (FIR hemodynamic delays, contiguous split, scoring, noise ceiling) |
| `features/` | LLM activation extraction (dense now; SAE next) |
| `controls/` | Baselines & statistics |
| `taxonomy/` | SAE feature labeling + inter-rater agreement |
| `configs/` | One file per configuration |
| `scripts/` | Entry points (`smoke_test.py`, `download_data.sh`) |
| `Notes/` | Plain-English breakdown + literature summaries + `Vocab/` |
| `Plans/` | Week-by-week execution plan (4 phases, 3 gates) |
| `PROGRESS.md` | Plain-English progress overview |
| `NEXT_STEPS.md` | Laptop-scoped plan with time estimates |
| `BUILD_LOG.md` | Detailed technical build log |
| `CLAUDE.md` | Protected project context |

## Data (not in repo)
fMRI = OpenNeuro **ds003020** (LeBel/Huth). Preprocessed matrices + TextGrids live **outside** the repo in `~/brain-llm-data/`. Never commit raw data (see `.gitignore`).

## Setup
```bash
uv venv --python 3.12 .venv       # 3.14 breaks numpy
uv pip install -e .               # deps from pyproject.toml
.venv/bin/python scripts/smoke_test.py   # validates the pipeline end-to-end
```

## Status
Gate A passed: GPT-2 layer 9 dense baseline reproduced on subject UTS01 (mean test corr 0.22, top-1% 0.53). See `PROGRESS.md` and `NEXT_STEPS.md`.
