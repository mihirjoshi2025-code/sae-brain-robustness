"""SAE availability check (NEXT_STEPS T2): residual SAEs per model in SAELens.

For every model in the SAELens pretrained registry, collect residual-stream SAEs,
their widths and layer coverage, and apply the model-3 rule: >= 2 widths at a
residual hook in each of the early / middle / late thirds of the network, and
<= 8 GB in bf16 (params from the HF config). Widths are parsed from the release
/ sae_id / Neuronpedia id text (e.g. "16k", "32x", "2pow14"), so treat the table
as a pointer to verify with SAE.from_pretrained, not as ground truth.

Usage:
  python scripts/check_saes.py [--all]
"""
import argparse
import re
from collections import defaultdict

from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory

N_LAYERS = {"gpt2-small": 12, "gemma-2-2b": 26, "pythia-70m-deduped": 6,
            "meta-llama/Llama-3.1-8B": 32, "Qwen/Qwen3-1.7B-Base": 28,
            "Qwen/Qwen3.5-2B-Base": 24, "gemma-2-9b": 42,
            "google/gemma-3-1b-pt": 26, "google/gemma-3-4b-pt": 34}
PARAMS_B = {"gpt2-small": 0.124, "gemma-2-2b": 2.6, "pythia-70m-deduped": 0.07,
            "meta-llama/Llama-3.1-8B": 8.0, "Qwen/Qwen3-1.7B-Base": 1.7,
            "Qwen/Qwen3.5-2B-Base": 2.0, "gemma-2-9b": 9.2,
            "google/gemma-3-1b-pt": 1.0, "google/gemma-3-4b-pt": 4.3}
D_MODEL = {"gpt2-small": 768, "meta-llama/Llama-3.1-8B": 4096, "pythia-70m-deduped": 512}


def width_of(text: str, model: str):
    t = text.lower()
    if m := re.search(r"2pow(\d+)", t):
        return 2 ** int(m.group(1))
    if m := re.search(r"(\d+)x\b|_(\d+)x", t):
        return int(m.group(1) or m.group(2)) * D_MODEL.get(model, 0) or None
    if m := re.search(r"(\d+(?:\.\d+)?)([km])\b|(\d+)([km])[-_/]", t):
        n, u = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
        return int(float(n) * (1024 if u == "k" else 1024 ** 2))
    return None


def layer_of(text: str):
    if m := re.search(r"(?:blocks\.|layer_?|resid_post_layer_|^l)(\d+)", text):
        return int(m.group(1))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="list every model, not just rule results")
    args = ap.parse_args()
    cov = defaultdict(lambda: defaultdict(set))  # model -> layer -> {width}
    for rel, v in get_pretrained_saes_directory().items():
        if re.search(r"transcoder|clt|att|mlp|hook-z|lxa|lxm|-it\b|instruct", rel):
            continue
        for sid, path in v.saes_map.items():
            if re.search(r"attn|mlp|hook_z", sid):
                continue
            npid = (v.neuronpedia_id or {}).get(sid) or ""
            w = width_of(f"{rel} {sid} {path} {npid}", v.model)
            L = layer_of(sid)
            if w and L is not None:
                cov[v.model][L].add(w)

    print(f"{'model':28s} {'layers w/ >=2 widths':24s} thirds(E/M/L) bf16GB rule")
    for model, layers in sorted(cov.items()):
        n = N_LAYERS.get(model)
        two = sorted(L for L, ws in layers.items() if len(ws) >= 2)
        if n:
            thirds = [any(i * n / 3 <= L < (i + 1) * n / 3 for L in two) for i in range(3)]
        else:
            thirds = [None] * 3
        gb = PARAMS_B.get(model, float("nan")) * 2
        ok = all(thirds) and gb <= 8 if n else "n_layers?"
        if args.all or two:
            rng = f"{two[0]}-{two[-1]} ({len(two)})" if two else "-"
            th = "".join("?" if t is None else ("Y" if t else "n") for t in thirds)
            widths = sorted({w for ws in layers.values() for w in ws})
            print(f"{model:28s} {rng:24s} {th:13s} {gb:6.1f} {ok}  widths={[f'{w // 1024}k' for w in widths]}")


if __name__ == "__main__":
    main()
