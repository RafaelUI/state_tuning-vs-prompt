#!/usr/bin/env python3
"""Сводка пункта 1 (30.10): провокации личности по всем плечам. Запуск из корня: python3 experiments/1030_provocations_control/provsum.py
Доля «чужой личности» (foreign) и «своего имени» (own) по 20 провокациям; наводящие (да/нет: «Ты X?», «Are you X?»,
«Признайся…») против открытых («Кто тебя создал?», «Which model…», «Как тебя зовут на самом деле?»).
Разность «чужая» состояние − промпт и состояние − без промпта: двухуровневый бутстрэп (сиды × вопросы, 95 %)."""
import json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prov

P = "experiments/1030_provocations_control/prov"
models = ["g1d_0p4b_bf16", "g1d_0p4b_red", "g1d_0p4b_comp", "g1d_1p5b_bf16", "g1d_1p5b_red", "g1d_1p5b_comp",
          "g1j_1p5b_comp", "g1j_2p9b_comp", "orion_g1d_0p4b_bf16", "orion_g1d_1p5b_bf16"]
rng = np.random.default_rng(0)
B = 10000


def mats(path, own):
    """[вопросы] доли own и foreign по сэмплам."""
    _, labs, rows = prov.arm_rates(path, own)
    qs = [r["question"] for r in rows]
    f = np.array([np.mean([x == "foreign" for x in l]) for l in labs])
    o = np.array([np.mean([x == "own" for x in l]) for l in labs])
    return qs, o, f


def boot(a, b, idx):
    a, b = a[:, idx], b[:, idx]; n = len(idx)
    m = a.mean() - b.mean()
    qi = rng.integers(0, n, size=(B, n)); sa = rng.integers(0, a.shape[0], size=(B, a.shape[0])); sb = rng.integers(0, b.shape[0], size=(B, b.shape[0]))
    d = np.array([a[sa[k]].mean(0)[qi[k]].mean() - b[sb[k]].mean(0)[qi[k]].mean() for k in range(B)])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return f"{m:+.2f} {{{lo:+.2f}; {hi:+.2f}}}"


print("# Пункт 1: провокации личности (30.10)\n")
print("20 провокаций (w1030/provocations.txt; с обучением не пересекаются) × 10 сэмплов, temp 1.0, top-p 0.6, 120 токенов. "
      "Разметка — w1030/prov.py (ручная сверка ~230 ответов: ~5 % ошибок). «Чужая» — присваивает чужую модель/создателя; "
      "«своё» — называет своё имя без чужого. Наводящие — 12 вопросов да/нет, открытые — 8. Сиды состояний — 7/8/9 через «/».\n")
print("| модель | плечо | своё | чужая (все) | чужая, наводящие | чужая, открытые |")
print("|---|---|---|---|---|---|")
diffs = []
for m in models:
    own = "orion" if m.startswith("orion") else "neuron"
    d = f"{P}/{m}"
    base = m.replace("orion_", "")
    arms = {"без промпта": [f"{P}/{base}/none.json"], "промпт": [f"{d}/inline.json"],
            "c4": [f"{d}/c4_s{s}.json" for s in (7, 8, 9)], "c": [f"{d}/c_s{s}.json" for s in (7, 8, 9)]}
    arms = {k: v for k, v in arms.items() if all(os.path.exists(p) for p in v)}
    M = {}
    for k, ps in arms.items():
        res = [mats(p, own) for p in ps]
        qs = res[0][0]
        M[k] = (np.stack([r[1] for r in res]), np.stack([r[2] for r in res]))
    lead = [i for i, q in enumerate(qs) if prov.YESNO_Q.search(q)]
    opn = [i for i in range(len(qs)) if i not in lead]
    for k, (o, f) in M.items():
        sd = lambda x, idx=None: " / ".join(f"{v:.2f}" for v in (x[:, idx].mean(1) if idx else x.mean(1))) if x.shape[0] > 1 else f"{(x[:, idx] if idx else x).mean():.2f}"
        print(f"| {m} | {k} | {sd(o)} | {sd(f)} | {sd(f, lead)} | {sd(f, opn)} |")
    for a, b in (("c4", "промпт"), ("c4", "без промпта")):
        if a in M and b in M:
            diffs.append((m, f"{a} − {b}", boot(M[a][1], M[b][1], list(range(len(qs)))), boot(M[a][1], M[b][1], lead), boot(M[a][1], M[b][1], opn)))
print("\n## Разности доли «чужой личности» {с×в}\n")
print("| модель | разность | все | наводящие | открытые |\n|---|---|---|---|---|")
for r in diffs:
    print("| " + " | ".join(r) + " |")
