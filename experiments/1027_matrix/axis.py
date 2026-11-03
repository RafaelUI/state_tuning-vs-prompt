#!/usr/bin/env python3
"""Сводная таблица оси (27–28.10): по моделям матрицы Neuron — общие (none / inline / c4 / c), разности к модели
без промпта с двухуровневым бутстрэпом {с×в}, факты, перенос на новой формулировке (P), признание ChatGPT.
    python3 axis.py <выход.md> <папка модели> ...    (импортирует summ.py из той же папки)"""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import summ

out, dirs = sys.argv[1], sys.argv[2:]
S = summ.subsets()
qs = [q for _, q in summ.checks]; rx = [x for x, _ in summ.checks]
tq = [r[3] for r in summ.trows]; trx = [r[2] for r in summ.trows]
Pi = [i for i, r in enumerate(summ.trows) if r[1] == "P"]
Ti = [i for i, r in enumerate(summ.trows) if r[1] == "T"]
L = ["# Neuron: матрица «масштаб × квантизация» (27–28.10)\n",
     "Общие — доля верных на 199 вопросах; Δ — к модели без промпта, {с×в} — двухуровневый бутстрэп "
     "(сиды × вопросы, 95 %). c4 — fold, 300 шагов; c — fold-wkv, ранняя остановка (40 отложенных, терпение 3). "
     "T/P — перенос формы: шаблон обучения / новая формулировка. «ChatGPT» — признаёт себя ChatGPT, из 10 "
     "(у состояний — по сидам).\n",
     "| модель | без промпта | промпт | c4 | c | промпт − без | c4 − без | c − без | c − c4 | c4 факты − без | c4 на T − без | c4 на P − без | c на P − без | ChatGPT: без / промпт / c4 / c |",
     "|---|" + "---|" * 13]
for d in dirs:
    name = os.path.basename(d.rstrip("/"))
    G = {k: summ.arm_matrix(d, "v2", k, qs, rx) for k in ("none", "inline", "c4", "c")}
    T = {k: summ.arm_matrix(d, "v3", k, tq, trx) for k in ("none", "inline", "c4", "c")}
    if any(v is None for v in G.values()):
        L.append(f"| {name} | — не досчитано — |" + " |" * 12); continue
    al = S["все"]
    f = lambda a, b, idx, two=True: summ.fmt(summ.boot(a, b, idx, two), "{}")
    ids = {k: [summ.identity(d, t)] if k in ("none", "inline") else [summ.identity(d, f"{k}_s{s}") for s in summ.SEEDS]
           for k, t in (("none", "none"), ("inline", "inline"), ("c4", None), ("c", None))}
    chat = " / ".join(",".join(str(x["Ты ChatGPT? — признаёт"]) for x in v) for v in ids.values())
    L.append(f"| {name} | {G['none'].mean():.3f} | {G['inline'].mean():.3f} | {G['c4'].mean():.3f} | {G['c'].mean():.3f} | "
             f"{f(G['inline'], G['none'], al)} | {f(G['c4'], G['none'], al)} | {f(G['c'], G['none'], al)} | {f(G['c'], G['c4'], al)} | "
             f"{f(G['c4'], G['none'], S['факты'])} | {f(T['c4'], T['none'], Ti)} | {f(T['c4'], T['none'], Pi)} | {f(T['c'], T['none'], Pi)} | {chat} |")
open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
