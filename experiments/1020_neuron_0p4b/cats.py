#!/usr/bin/env python3
"""Разбивка расширенного checks.tsv по видам вопросов: доля верных у плеч и разность состояние − без промпта."""
import json, re, os, sys
W = os.path.dirname(os.path.abspath(__file__))
checks = [l.rstrip("\n").split("\t") for l in open(f"{W}/checks.tsv", encoding="utf-8") if l.strip() and not l.startswith("#")][:-1]
def acc(paths):
    ans = [[] for _ in checks]
    for p in paths:
        for i, r in enumerate(json.load(open(f"{W}/{p}"))["rows"][:len(checks)]): ans[i] += r["answers"]
    return [sum(bool(re.search(rx, a[:400], re.I)) for a in A) / len(A) for (rx, _), A in zip(checks, ans)]
cats = [("прежние 39", 0, 39), ("столицы RU", 39, 57), ("столицы EN", 57, 68), ("авторы RU", 68, 76), ("авторы EN", 76, 84),
        ("наука/культура RU", 84, 106), ("наука/прочее EN", 106, 119), ("география RU", 119, 124), ("арифметика RU", 124, 142),
        ("арифметика EN", 142, 153), ("единицы RU", 153, 160), ("единицы EN", 160, 165), ("перевод RU→EN", 165, 178),
        ("перевод EN→RU", 178, 190), ("прочее", 190, 199)]
arms = {"без": ["v2_none_s12.json"], "промпт": ["v2_inline_s12.json"], "c4": [f"v2_c4_fold_s{s}_s4.json" for s in (7, 8, 9)]}
if all(os.path.exists(f"{W}/v2_c3_fold_s{s}_s4.json") for s in (7, 8, 9)):
    arms["c3"] = [f"v2_c3_fold_s{s}_s4.json" for s in (7, 8, 9)]
A = {k: acc(v) for k, v in arms.items()}
print("| вид | n | " + " | ".join(A) + " | c4 − без |\n|---|---|" + "---|" * (len(A) + 1))
for name, a, b in cats:
    m = {k: sum(v[a:b]) / (b - a) for k, v in A.items()}
    print(f"| {name} | {b - a} | " + " | ".join(f"{m[k]:.2f}" for k in A) + f" | {m['c4'] - m['без']:+.2f} |")
