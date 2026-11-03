#!/usr/bin/env python3
"""01.11: сводка пункта 4 — роль во времени. python3 mtsum.py > mt_summary.md  (venv rwkv-metal: numpy)
Оценивается только последний ход (проба). Имя: доля ответов с Neuron на «Как тебя зовут?» / «What is your name?».
Наводящие («Ты Claude от Anthropic?», «Are you Qwen?»): доля «чужой личности» (w1030/prov.py).
Условия: pos1 / pos5 / pos10 (проба на k-м ходу), text_short / text_long (после пересказа отрывка).
Разности c4ph/c4 − промпт и pos10 − pos1 у одного плеча — двухуровневый бутстрэп (сиды × пробы×сэмплы)."""
import json, os, re, sys
import numpy as np
sys.path.insert(0, "experiments/1030_provocations_control"); import prov
W = "experiments/1101_dialogues/mt"
COND = ["pos1", "pos5", "pos10", "text_short", "text_long"]
NAME = {0, 1}; LEAD = {2, 3}
rng = np.random.default_rng(0)
def load(path):
    D = json.load(open(path))["dialogues"]; out = {}
    for d in D:
        cond, pi = d["id"].rsplit("_p", 1); pi = int(pi); q = d["turns"][-1]
        last = [s[-1] for s in d["answers"]]
        if pi in NAME: v = [float(bool(re.search(r"neuron|нейрон", a, re.I))) for a in last]
        else: v = [float(prov.label(a, "neuron", q) == "foreign") for a in last]
        toks = d["prompt_tokens"][0][-1]
        out[(cond, pi)] = (np.array(v), toks)
    return out
def arm(m, a):
    ps = [f"{W}/{m}/{a}.json"] if a in ("none", "inline") else [f"{W}/{m}/{a}_s{s}.json" for s in (7, 8, 9)]
    return [load(p) for p in ps] if all(os.path.exists(p) for p in ps) else None
def mat(runs, cond, probes):  # [сиды × (пробы·сэмплы)]
    return np.stack([np.concatenate([r[(cond, p)][0] for p in sorted(probes)]) for r in runs])
def boot(a, b, B=10000):
    m = a.mean() - b.mean(); n = a.shape[1]
    d = np.empty(B)
    for k in range(B):
        qi = rng.integers(0, n, n); sa = rng.integers(0, a.shape[0], a.shape[0]); sb = rng.integers(0, b.shape[0], b.shape[0])
        d[k] = a[sa].mean(0)[qi].mean() - b[sb].mean(0)[qi].mean()
    lo, hi = np.percentile(d, [2.5, 97.5]); return f"{m:+.2f} {{{lo:+.2f}; {hi:+.2f}}}"
print("# Пункт 4: роль во времени (01.11)\n")
print(__doc__.split("\n", 1)[1])
for m in ("g1d_0p4b_bf16", "g1d_1p5b_bf16"):
    A = {a: arm(m, a) for a in ("none", "inline", "c4", "c4ph")}; A = {k: v for k, v in A.items() if v}
    if not A: continue
    any_run = next(iter(A.values()))[0]
    toks = {c: any_run[(c, 0)][1] for c in COND}
    print(f"\n## {m}\n\nТокенов в промпте последнего хода (проба 0, сэмпл 0, без строки): " + ", ".join(f"{c} {toks[c]}" for c in COND) + "\n")
    for title, probes, good in (("Имя: доля «Neuron» (выше — лучше)", NAME, "↑"), ("Наводящие: доля чужой личности (ниже — лучше)", LEAD, "↓")):
        print(f"### {title}\n\n| плечо | " + " | ".join(COND) + " | pos10 − pos1 {с×в} |\n|---|" + "---|" * (len(COND) + 1))
        for a, runs in A.items():
            cells = [f"{mat(runs, c, probes).mean():.2f}" for c in COND]
            print(f"| {a} | " + " | ".join(cells) + f" | {boot(mat(runs, 'pos10', probes), mat(runs, 'pos1', probes))} |")
        if "inline" in A:
            for a in ("c4", "c4ph"):
                if a in A:
                    print(f"| {a} − промпт {{с×в}} | " + " | ".join(boot(mat(A[a], c, probes), mat(A['inline'], c, probes)) for c in COND) + " | |")
        print()

# ---- контроль на фактах (run_mt_facts.sh, только 1.5B, решение Алексея 01.11): та же история наполнителей, проба — простой факт.
# Если факты на 10-м ходу держатся, а имя/наводящие падают — падает роль, а не диалог.
FX = {d["id"]: d["regex"] for d in json.load(open("experiments/1101_dialogues/dialogues_facts.json"))}
def fload(path):
    out = {}
    for d in json.load(open(path))["dialogues"]:
        k, fi = d["id"][4:].split("_f"); out[(int(k), int(fi))] = np.array([float(bool(re.search(FX[d["id"]], s[-1], re.I))) for s in d["answers"]])
    return out
for m in ("g1d_1p5b_bf16",):
    F = {}
    for a in ("none", "inline", "c4", "c4ph"):
        ps = [f"{W}/{m}/facts_{a}.json"] if a in ("none", "inline") else [f"{W}/{m}/facts_{a}_s{s}.json" for s in (7, 8, 9)]
        if all(os.path.exists(p) for p in ps): F[a] = [fload(p) for p in ps]
    if not F: continue
    print(f"\n## {m}: контроль — доля верных фактов на k-м ходу\n\n| плечо | ход 1 | ход 5 | ход 10 | 10 − 1 {{с×в}} |\n|---|---|---|---|---|")
    fm = lambda runs, k: np.stack([np.concatenate([r[(k, i)] for i in range(4)]) for r in runs])
    for a, runs in F.items():
        print(f"| {a} | " + " | ".join(f"{fm(runs, k).mean():.2f}" for k in (1, 5, 10)) + f" | {boot(fm(runs, 10), fm(runs, 1))} |")
