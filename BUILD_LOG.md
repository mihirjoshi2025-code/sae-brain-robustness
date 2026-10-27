# Build Log (technical details)

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

## Day 2 — DENSE BASELINE REPRODUCED (Gate A ✅)
**`scripts/run_gate_a.py` — GPT-2 layer 9 dense features → UTS01 BOLD.**
- Train: 10 stories (3,505 TRs). Test: `wheretheressmoke` (mean of 10 repeats).
- **mean test corr = 0.220** (top-10k voxels), **top-1% = 0.527**, 31 s. Consistent with published LeBel/Huth GPT-2 encoding.
- New modules: `pipeline/data.py` (BOLD load, repeats, TextGrid parser — handles BOTH long + chronological Praat formats), `pipeline/downsample.py` (Lanczos word→TR), `features/word_features.py` (sliding-window contextual per-word activations).
- Conventions locked: TR=2.0s; stimulus grid = n_bold+10; trim 5 each end; word tier = item[2]/tier 2; filter {NS}/sp.

### TODO / refinements (all closed on Day 3)
- ~~Noise ceiling: Spearman-Brown (N=10)~~ ✓
- ~~Sweep layers and FREEZE before SAE results~~ ✓
- ~~Log runs to a results table (CSV)~~ ✓

## Day 3 — Gate A tidy-up, layers frozen, SAE path (Oct 25)

**Noise ceiling (Step 1).** `pipeline/encoding.py::noise_ceiling_corr` → single-repeat mean pairwise r, Spearman-Brown to the 10-repeat average, sqrt → correlation-scale ceiling. UTS01 top-10k: single r 0.076 → ceiling **0.645**. Gate A L9: corr 0.223, **normalized 0.346** (old corr/r ≈ 2.9 was meaningless).

**Results table.** `pipeline/results.py::log_run` → `results/runs.csv`, fixed columns (unknown keys raise; old CSVs auto-migrated when columns are added). Records git commit (+`-dirty`), seed, voxel selection, alphas, delays, scores, ceiling, p-value columns.

**Shared code.** `pipeline/lebel.py` = one scoring path for every script (`tr_features`, `fit_score`, `fit_score_kernel`, `summarize`). FIR delays now applied **per story** (lags no longer bleed across story boundaries; L9 0.220 → 0.223). `features/cache.py`: word-level activations cached per model × layer × dataset × feature set in `~/brain-llm-data/activations/*.h5`.

**Layer sweep + freeze (Step 2).** `scripts/sweep_layers.py`: 5-fold story-level CV inside the 10 training stories (test story never used), rule fixed before running: best = max CV mean corr, early = round(0.25·(n−1)), late = n−1. Refuses to re-pick once frozen. GPT-2 CV: L0 0.090 → rises → **L6/L7 0.1221** (L7 higher at full precision) → L11 0.107. **Frozen: best 7, early 3, late 11** (`configs/layers.yaml`). Test story after freeze: L3 0.200 / **L7 0.225 (norm 0.348)** / L11 0.201.

**Bug fixed: HF last hidden state is post-LayerNorm.** `hidden_states[-1]` has `ln_f` applied (Gemma too), so "layer 11" ≠ `blocks.11.hook_resid_post` where SAEs attach. `word_activations` now reads each block's raw output with a forward hook. L0–L10 bit-identical; L11 recached. L11 CV 0.1085 → 0.1069; choice unchanged. Old L11 rows marked SUPERSEDED in the CSV; erratum in `configs/layers.yaml`.

**SAE coverage (Step 2).** sae-lens 6.54.12 installed (pulled transformer-lens 4.3 → 3.9; unused). `configs/models.yaml`:
- GPT-2: OpenAI v5 TopK resid_post **32k + 128k**, all 12 layers (same recipe at both widths).
- Gemma 2 2B: Gemma Scope res canonical **16k + 65k**, all 26 layers. Blocked: no HF token / Gemma license.
- Third model: open. Pythia-70m (SAEBench 4k/16k, layers 3–4 only), Llama 3.1 8B (Llama Scope 32k/128k, all layers; heavy), Qwen3-1.7B (one width only).

**SAE features (Step 3).** `features/sae.py`: SAELens `SAE.from_pretrained`, encode cached dense word activations, cache CSR codes per model × layer × dataset × width (`..._sae32k.h5`), FVU stored per story. GPT-2 L7 32k on `avatar`: L0 = 32 (TopK), FVU 0.256, 17.6% of features ever active.

**Kernel solver.** `pipeline/downsample.py` now builds a (T × N) Lanczos matrix (identical to old loop, max diff 0.0) so sparse codes downsample as sparse. `pipeline/lebel.py::fit_score_kernel`: same ridge on z-scored delayed features, solved in the dual with a precomputed linear kernel built from sparse matrices (mean/sd correction done algebraically, so 128k × 4 delays is never densified). **Check: dense L7 primal vs kernel, max |Δr| = 6e-7 over 10k voxels.** Dense and SAE results therefore come from the same model.

## Day 2 (original plan) — reproduce the dense baseline on real fMRI
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

**Controls (T6, built by a helper agent; random dictionary added by me).** `scripts/run_controls.py --frozen`, GPT-2 dense, test story, mean r / normalized / frac voxels FDR-sig:
L7 dense .225/.348/.79 · untrained .093/.144/.23 · random_matched (column shuffle) .059/.091/.13 · word_rate .069 · sentence_position .074 (pause ≥ 0.5 s = boundary, D13) · surprisal .077. Block permutation (10-TR blocks, 1000 perms) p = .001 for all; shuffled-target null gives 5–9% voxels at p<.05, 0 after FDR. `controls/random_dictionary.py` = T6 spec (ReLU(dense·R) top-k by real L0, variance spectrum matched; mean L0 31.98 vs 32).

**Alpha CV changed to story-level folds** (`lebel.story_folds`, `cv="story"` default for both solvers; `cv` column in runs.csv; blank = old KFold(5)). Parity (`scripts/test_kernel_parity.py`, L9): same-alpha voxels max |Δr| 6.4e-7, 1/10000 voxel flips alpha on a near-tied CV score (|Δr| 0.011 there), peak RSS 2.9 GB → PASS under the corrected acceptance in NEXT_STEPS T4.

**T2 model lock.** `scripts/check_saes.py` applies the model-3 rule. Passing: gpt2-small, gemma-2-2b, google/gemma-3-1b-pt (Gemma Scope 2 res-all, 16k + 262k at all 26 layers). Rejected: Llama-3.1-8B / gemma-3-4b / gemma-2-9b (> 8 GB bf16), Pythia-70m (SAEBench 4k/16k at L3–4 only), Qwen3-1.7B (one width). Caveat: models 2 and 3 are both Gemma.

**Claim 1 (T5), GPT-2, pre-registered pipeline — FAILS at every layer and size.** `scripts/run_claim1.py`; z-score every alive column, kernel solver, story folds; ratio = norm_SAE / norm_dense with 10-TR block-bootstrap 95% CI; holds if ci_low ≥ 0.9.

| layer | dense r | SAE 32k r | ratio [CI] | randdict 32k | SAE 128k r | ratio [CI] | randdict 128k |
|---|---|---|---|---|---|---|---|
| 3 | .200 | .006 | .03 [.005,.22] | .050 | .015 | .08 [−.001,.31] | .021 |
| 7 | .225 | −.001 | −.003 [−.08,.16] | .013 | .013 | .06 [.01,.10] | .037 |
| 11 | .200 | .022 | .11 [.07,.29] | .071 | .040 | .20 [.12,.30] | .094 |

SAE quality all pass (`purpose=sae_check`): FVU .25–.31 (< 0.4), mean L0 = 32 (TopK k=32, as published). Not a hook bug.
**FLAG (T6 sanity ordering): SAE < random dictionary at every layer/size.** Not "fixed", per spec. Likely mechanism: ~80% of alive SAE columns fire in < 1% of training TRs; per-column z-scoring inflates those rare columns until they dominate the kernel. Quick check (L3 32k, top-2000 voxels): the 3,000 most frequent SAE columns alone give r = .27 (kernel = primal on that subset, |Δ| 4e-6). L11 random dictionary uses far fewer columns (5.7k vs 51k alive), consistent with rarity driving the gap. Follow-up `scripts/diag_sae_normalization.py` (POST-HOC, labeled `purpose=diagnostic_normalization`): column vs ≥1%-frequency filter vs global scale, same for dense/SAE/randdict, best layer only.
Note: 4 `claim1_sae_vs_dense` rows with blank `cv` and no CI are from an aborted first run (KFold alphas); superseded by the rows with `cv=story5`.

**Normalization diagnostic (POST-HOC, L7, `purpose=diagnostic_normalization`).** mean r (normalized):
| features | column (pre-reg) | freq ≥1% (cols kept) | global scale |
|---|---|---|---|
| dense | .225 (.348) | .225 (.348) | .205 (.318) |
| SAE 32k | −.001 | **.173 (.268)**, 16.5k of 62k cols | .149 (.230) |
| randdict 32k | .013 | .153, 19.1k cols | .166 |
| SAE 128k | .013 | **.170 (.263)**, 14.4k of 117k cols | .145 (.224) |
| randdict 128k | .037 | .157, 25.8k cols | .152 |
Takeaways: (1) rare-column inflation explains the near-zero pre-registered SAE scores. (2) Under either alternative SAE ≈ 70–77% of dense (point estimates, no CI): still below the 0.9 bar, so the Claim 1 verdict is "fails" in all three variants, but by very different margins. (3) Whether SAE beats the random dictionary also flips: SAE wins with the frequency filter (.173 vs .153), loses with global scaling (.149 vs .166). Both are evaluation-choice sensitivities worth reporting.

## Day 4 (Oct 27): Gemma layers frozen (T9 part 1)

`scripts/overnight_gemma_sweeps.sh` (caffeinate; resumable from runs.csv folds) ran `sweep_layers.py` for both Gemma models on the 25-story UTS01 training set, story-level 5-fold CV, top 10,000 voxels, `cv=story5`. Gemma 2 finished ~07:35, Gemma 3 at 10:37 (exit 0). Activations read from the raw decoder-block output via hooks, BOS prepended per window, bf16.

| model | best | early | late | peak CV r | layer-25 CV r |
|---|---|---|---|---|---|
| gemma-2-2b | 13 | 6 | 25 | .1457 | .1198 |
| gemma-3-1b-pt | 10 | 6 | 25 | .13764 | .1174 |

Gemma 3 near-tie: L10 .13764 vs L11 .13756 (5-fold means from runs.csv); argmax rule picks 10, no override. GPT-2 layers stay frozen from the 10-story sweep (not re-picked). Written to `configs/layers.yaml`; no SAE result for either Gemma model exists yet.
