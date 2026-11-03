#!/usr/bin/env python3
"""Перенос формы (22.10): transfer.tsv — 40 единиц (слово на перевод / пример), у каждой две формулировки:
T — шаблон обучающих фактов, P — новая. Доля верных на вопрос по всем ответам плеча; прибавка плеча к модели без
промпта отдельно на T и на P; «перенос» — (прибавка на P) / (прибавка на T); разность прибавок T − P — бутстрэп по
единицам (10 000). Плечи: v3_none_s12, v3_inline_s12, v3_<состояние>_s{7,8,9}_s4.
    python3 transfer.py [папка] [префикс состояния, по умолчанию c4_fold]"""
import json, re, os, sys, random
W = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
st = sys.argv[2] if len(sys.argv) > 2 else "c4_fold"
rows = [l.rstrip("\n").split("\t") for l in open(f"{W}/transfer.tsv", encoding="utf-8") if l.strip() and not l.startswith("#")]
def acc(paths):
    ans = [[] for _ in rows]
    for p in paths:
        for i, r in enumerate(json.load(open(f"{W}/{p}"))["rows"]):
            assert r["question"] == rows[i][3]; ans[i] += r["answers"]
    return [sum(bool(re.search(rx, a[:400], re.I)) for a in A) / len(A) for (_, _, rx, _), A in zip(rows, ans)]
arms = {"без промпта": ["v3_none_s12.json"], "промпт": ["v3_inline_s12.json"],
        "состояние": [f"v3_{st}_s{s}_s4.json" for s in (7, 8, 9)]}
A = {k: acc(v) for k, v in arms.items() if all(os.path.exists(f"{W}/{p}") for p in v)}
units = sorted({r[0] for r in rows})
idx = {(r[0], r[1]): i for i, r in enumerate(rows)}
groups = {"все": units, "перевод RU→EN": [u for u in units if u.startswith("tr")], "перевод EN→RU": [u for u in units if u.startswith("te")],
          "арифметика RU": [u for u in units if u.startswith("ar")], "арифметика EN": [u for u in units if u.startswith("ae")]}
rng = random.Random(0)
def boot(d):
    m = sum(d) / len(d)
    bs = sorted(sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(10000))
    return f"{m:+.3f} [{bs[250]:+.3f}; {bs[9750]:+.3f}]"
print("| плечо | группа | T (шаблон обучения) | P (новая формулировка) |\n|---|---|---|---|")
for k, a in A.items():
    for g, us in groups.items():
        print(f"| {k} | {g} | {sum(a[idx[u, 'T']] for u in us) / len(us):.3f} | {sum(a[idx[u, 'P']] for u in us) / len(us):.3f} |")
if "без промпта" in A:
    print("\n| разница | группа | на T | на P | (T − P) прибавок |\n|---|---|---|---|---|")
    for k in [x for x in A if x != "без промпта"]:
        for g, us in groups.items():
            dT = [A[k][idx[u, "T"]] - A["без промпта"][idx[u, "T"]] for u in us]
            dP = [A[k][idx[u, "P"]] - A["без промпта"][idx[u, "P"]] for u in us]
            print(f"| {k} − без промпта | {g} (n={len(us)}) | {boot(dT)} | {boot(dP)} | {boot([x - y for x, y in zip(dT, dP)])} |")
