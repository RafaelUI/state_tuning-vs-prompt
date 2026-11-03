#!/usr/bin/env python3
"""02.11: сводка кривой по объёму данных (run_curve.sh). venv rwkv-metal: python curvesum.py > curve_summary.md
Строгая точность на 150 статьях ~400 токенов; N = 25/50/125 — curve/, N = 250 — основной прогон (mid_cg_s*, mid_lora_s*).
Разность состояние − LoRA — двухуровневый бутстрэп {с×в} (сиды × статьи, 95 %, 10 000 повторов)."""
import json, os, re
import numpy as np
C = "experiments/1101_classification"
LAB = ["спорт", "экономика", "наука", "культура", "путешествия"]
rx = re.compile(r"^[\s*«\"]*(" + "|".join(LAB) + r")[\s*»\".!]*$", re.I)
gold = [l.rstrip("\n").split("\t")[0] for l in open(f"{C}/test_labels.tsv", encoding="utf-8") if not l.startswith("#") and l.split("\t")[1] == "mid"]
def mat(paths):
    if not all(os.path.exists(p) for p in paths): return None
    out = []
    for p in paths:
        R = json.load(open(p))["rows"]
        out.append([np.mean([bool((m := rx.match(a.strip()))) and m.group(1).lower() == g for a in r["answers"]]) for r, g in zip(R, gold)])
    return np.array(out)
def files(n, a):
    if n == 250: return [f"{C}/g1d_0p4b_bf16/mid_{a}_s{s}.json" for s in (7, 8, 9)]
    return [f"{C}/curve/g1d_0p4b_bf16/mid_n{n}_{a}_s{s}.json" for s in (7, 8, 9)]
rng = np.random.default_rng(0)
def boot(a, b, B=10000):
    m = a.mean() - b.mean(); n = a.shape[1]; d = np.empty(B)
    for i in range(B):
        q = rng.integers(0, n, n); sa = rng.integers(0, 3, 3); sb = rng.integers(0, 3, 3)
        d[i] = a[sa].mean(0)[q].mean() - b[sb].mean(0)[q].mean()
    lo, hi = np.percentile(d, [2.5, 97.5]); return f"{m:+.3f} {{{lo:+.3f}; {hi:+.3f}}}"
print("# Кривая по объёму данных: состояние против LoRA, g1d 0.4B (02.11)\n\n" + __doc__.split("\n", 1)[1] + "\n")
print("| статей (на рубрику) | состояние | сиды | LoRA r16 | сиды | состояние − LoRA {с×в} |\n|---|---|---|---|---|---|")
for n in (25, 50, 125, 250):
    S, L = mat(files(n, "cg")), mat(files(n, "lora"))
    sd = lambda x: " / ".join(f"{v:.3f}" for v in x.mean(1)) if x is not None else ""
    print(f"| {n} ({n // 5}) | {S.mean():.3f} | {sd(S)} | {L.mean():.3f} | {sd(L)} | {boot(S, L)} |" if S is not None and L is not None else f"| {n} ({n // 5}) | — | | — | | |")
