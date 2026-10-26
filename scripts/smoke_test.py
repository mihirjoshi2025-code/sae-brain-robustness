"""Smoke test: validate the encoding pipeline end-to-end BEFORE real fMRI.

Flow:
  1. Load GPT-2, extract dense activations for N text chunks at a middle layer.
  2. Build a SYNTHETIC voxel target that is a known linear function of the
     activations + noise  ->  the encoder MUST recover signal (sanity check).
  3. Build a RANDOM target with no relation  ->  the encoder MUST score ~0
     (control sanity check: no spurious signal from the pipeline itself).

If (2) scores high and (3) scores ~0, the pipeline mechanics are correct.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.extract import load_model, dense_activations, get_device  # noqa: E402
from pipeline.encoding import fit_score_encoding  # noqa: E402

SEED = 0
LAYER = 6          # a middle layer of gpt2 (12 layers)
N_CHUNKS = 400     # pretend-TRs
N_VOXELS = 50

rng = np.random.default_rng(SEED)
print(f"device: {get_device()}")

# --- 1. fake "stimulus": varied short sentences (stand-in for story TRs) ---
subjects = ["the dog", "a scientist", "the old ship", "her brother", "the city",
            "a quiet river", "the broken clock", "my teacher", "the red door", "a storm"]
verbs = ["ran across", "studied", "sailed past", "remembered", "lit up",
         "flowed through", "stopped near", "walked into", "opened onto", "swept over"]
objects = ["the field", "the problem", "the harbor", "the summer", "the night",
           "the valley", "the station", "the garden", "the sea", "the hills"]
chunks = []
for _ in range(N_CHUNKS):
    chunks.append(f"{rng.choice(subjects)} {rng.choice(verbs)} {rng.choice(objects)}.")

print(f"loading gpt2, extracting layer {LAYER} activations for {N_CHUNKS} chunks...")
bundle = load_model("gpt2")
X = dense_activations(bundle, chunks, layer=LAYER)
print(f"activations: {X.shape}  (T x d_model)")

# --- 2. synthetic target that depends on X with a hemodynamic-style lag ---
# BOLD lags, so make Y[t] depend on X[t-2]; the encoder's FIR delays (1..4)
# include d=2, so a correct pipeline recovers it.
LAG = 2
d = X.shape[1]
W = rng.standard_normal((d, N_VOXELS)).astype(np.float32)
Xlag = np.zeros_like(X)
Xlag[LAG:] = X[:-LAG]
signal = Xlag @ W
Y_signal = signal + 2.0 * rng.standard_normal(signal.shape).astype(np.float32)

# --- 3. pure-noise target (should NOT be predictable) ---
Y_noise = rng.standard_normal((N_CHUNKS, N_VOXELS)).astype(np.float32)

print("\nfitting encoder on SIGNAL target (should be HIGH)...")
r_sig = fit_score_encoding(X, Y_signal, seed=SEED)
print(f"  mean test corr = {r_sig['mean_score']:.3f}  "
      f"(features={r_sig['n_features']}, voxels={r_sig['n_voxels']}, "
      f"train={r_sig['n_train']}, test={r_sig['n_test']})")

print("fitting encoder on NOISE target (should be ~0)...")
r_noise = fit_score_encoding(X, Y_noise, seed=SEED)
print(f"  mean test corr = {r_noise['mean_score']:.3f}")

# --- verdict ---
ok = r_sig["mean_score"] > 0.3 and abs(r_noise["mean_score"]) < 0.1
print("\n" + ("PASS: pipeline recovers signal and rejects noise."
              if ok else "FAIL: check the pipeline."))
sys.exit(0 if ok else 1)
