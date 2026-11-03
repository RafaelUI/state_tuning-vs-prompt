#!/usr/bin/env python3
"""Сводка пункта 2 (31.10): сильный контроль — те же отфильтрованные ответы учителя, учатся веса (top4 / LoRA r16)
вместо начального состояния. Запуск из корня: python3 experiments/1030_provocations_control/ctlsum.py"""
import json, os, re, sys
import numpy as np
sys.path.insert(0, "experiments/1027_matrix"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import summ, prov

M = "experiments/1027_matrix/m"; C = "experiments/1030_provocations_control/ctl"; P = "experiments/1030_provocations_control/prov"
qs = [q for _, q in summ.checks]; rx = [x for x, _ in summ.checks]
tq = [r[3] for r in summ.trows]; trx = [r[2] for r in summ.trows]
Ti = [i for i, r in enumerate(summ.trows) if r[1] == "T"]; Pi = [i for i, r in enumerate(summ.trows) if r[1] == "P"]
S = summ.subsets()
f = lambda a, b, idx: summ.fmt(summ.boot(a, b, idx, True), "{}")


def prov_mat(paths):
    res = []
    for p in paths:
        _, labs, rows = prov.arm_rates(p, "neuron")
        res.append([np.mean([x == "foreign" for x in l]) for l in labs])
    return np.array(res)


def name_rate(paths):
    n = t = 0
    for p in paths:
        for r in json.load(open(p))["rows"][:4]:
            n += sum(bool(re.search(r"\bneuron\b|\bнейрон\b", a, re.I)) for a in r["answers"]); t += len(r["answers"])
    return n / t


print("# Пункт 2: сильный контроль (31.10)\n")
print("Neuron, данные c4 (тот же кэш учителя, что у матрицы). Состояние c4 — из матрицы; top4 — верхние 4 слоя + ln_out; "
      "LoRA — r16 на tmix всех слоёв. LR выбран по валидации (факты вне обучения), один сид; затем сиды 7/8/9. "
      "Δ — к модели без промпта, {с×в} — двухуровневый бутстрэп. «Чужая» — доля присвоения чужой личности на 20 провокациях.\n")
rows = [("g1d_0p4b_bf16", [("c4", "состояние c4", None), ("top", "top4 (lr 3e-5)", None)]),
        ("g1d_1p5b_bf16", [("c4", "состояние c4", None), ("lora", "LoRA r16 (lr 1e-4)", None), ("top", "top4 (lr 1e-5)", None)])]
print("| модель | плечо | обуч. параметров | общие | Δ общие | Δ факты | T − без | P − без | «Neuron» на 4 вопросах | чужая (провокации) |")
print("|---|---|---|---|---|---|---|---|---|---|")
params = {"g1d_0p4b_bf16": {"c4": "1.6 M", "top": "≈ 50 M"}, "g1d_1p5b_bf16": {"c4": "3.1 M", "lora": "≈ 6 M", "top": "≈ 200 M"}}
diffs = []
for m, arms in rows:
    none = summ.arm_matrix(f"{M}/{m}", "v2", "none", qs, rx); inl = summ.arm_matrix(f"{M}/{m}", "v2", "inline", qs, rx)
    tn = summ.arm_matrix(f"{M}/{m}", "v3", "none", tq, trx)
    pn = prov_mat([f"{P}/{m}/none.json"]); pi = prov_mat([f"{P}/{m}/inline.json"])
    print(f"| {m} | без промпта | — | {none.mean():.3f} | | | | | 0.00 | {pn.mean():.2f} |")
    print(f"| {m} | промпт | — | {inl.mean():.3f} | {f(inl, none, S['все'])} | {f(inl, none, S['факты'])} | | | "
          f"{name_rate([f'{M}/{m}/id_inline_s10.json']):.2f} | {pi.mean():.2f} |")
    mats = {}
    for k, label, _ in arms:
        d = f"{M}/{m}" if k == "c4" else f"{C}/{m}"
        G = summ.arm_matrix(d, "v2", k, qs, rx); T = summ.arm_matrix(d, "v3", k, tq, trx)
        pp = [f"{P}/{m}/c4_s{s}.json" for s in (7, 8, 9)] if k == "c4" else [f"{C}/{m}/prov_{k}_s{s}.json" for s in (7, 8, 9)]
        PV = prov_mat(pp)
        nr = name_rate([f"{d}/id_{k}_s{s}_s10.json" for s in (7, 8, 9)])
        mats[k] = (G, PV)
        seeds = " / ".join(f"{x:.3f}" for x in G.mean(1))
        print(f"| {m} | {label} | {params[m][k]} | {G.mean():.3f} ({seeds}) | {f(G, none, S['все'])} | {f(G, none, S['факты'])} | "
              f"{f(T, tn, Ti)} | {f(T, tn, Pi)} | {nr:.2f} | {PV.mean():.2f} |")
    for k in [a for a, _, _ in arms if a != "c4"]:
        diffs.append((m, f"c4 − {k}", f(mats['c4'][0], mats[k][0], S['все']), f(mats['c4'][0], mats[k][0], S['факты']),
                      summ.fmt(summ.boot(mats['c4'][1], mats[k][1], list(range(mats[k][1].shape[1])), True), "{}")))
print("\n## Состояние против контроля {с×в}\n")
print("| модель | разность | общие | факты | чужая (провокации) |\n|---|---|---|---|---|")
for r in diffs:
    print("| " + " | ".join(r) + " |")
