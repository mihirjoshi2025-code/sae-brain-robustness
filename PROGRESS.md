# Build Progress Log

## Day 1 (today)
**Task 1 — environment + pipeline scaffold + smoke test: DONE.**

- Repo scaffold: `pipeline/ features/ controls/ taxonomy/ configs/ scripts/`.
- Data root is OUTSIDE the vault: `~/brain-llm-data/` (preprocessed, textgrids, cache, activations).
- uv env pinned to **Python 3.12** at `.venv` (3.14 breaks numpy — Cornell lesson). Installed: torch 2.14 (MPS OK), himalaya, nibabel, transformers, numpy 2.5, scipy, sklearn, pandas, h5py.
- **TransformerLens 4.0 removed `HookedTransformer`** → switched dense extraction to HuggingFace `transformers` + `output_hidden_states` (stable). hidden_states[L+1] = residual stream after block L = where public SAEs attach. SAELens wiring deferred to the SAE step.
- `pipeline/encoding.py`: FIR hemodynamic delays, contiguous (blocked) split, z-score, himalaya RidgeCV, correlation scoring, noise-ceiling-from-repeats.
- `features/extract.py`: GPT-2 dense activations, mean-pooled per chunk.
- `scripts/smoke_test.py`: **PASS** — signal target corr **0.985**, noise target **0.002**. Pipeline recovers known signal, rejects noise. Confirms mechanics before real fMRI.

**Data download (background):** `scripts/download_data.sh` → one subject **UTS01** of ds003020:
- all 84 TextGrids (word timings) ✓
- test story `wheretheressmoke.hf5` (2GB, repeated → noise ceiling) — downloading
- 10 training stories (adollshouse avatar buck exorcism food legacy naked souls sweetaspie treasureisland)
- log: `~/brain-llm-data/download.log`

## Day 2 (next) — reproduce the dense baseline on real fMRI
1. **Parse TextGrids** (word tier = item 2): get (word, onset, offset) per story.
2. **Load preprocessed .hf5** → BOLD matrix (TR × voxels). Confirm TR (LeBel = 2.0s) and trimming convention.
3. **Align words → TRs**: extract GPT-2 activations per word, downsample word features to TR grid (Lanczos), trim front/back TRs to match BOLD.
4. **Fit dense encoding** per story; train on 10 stories, test on `wheretheressmoke`.
5. **Noise ceiling** from the repeats in `wheretheressmoke`.
6. Log to a results table. ← this is **Gate A** (dense baseline reproduced).

Then, if time: swap dense → SAE features (SAELens) for one config; add random-feature + untrained baselines.

## Open decisions / notes
- Model 1 = **gpt2** (small) for speed; has public SAEs for the SAE step.
- Need to confirm exact TR trimming (Huth pipeline trims ~5 front / ~10 back TRs). Check preprocessed .hf5 shape vs TextGrid duration to infer.
- Keep all .hf5 / weights OUT of the Obsidian vault.
