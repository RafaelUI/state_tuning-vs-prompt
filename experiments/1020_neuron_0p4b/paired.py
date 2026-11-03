#!/usr/bin/env python3
"""Парное сравнение общих способностей (checks.tsv, 39 вопросов без имени): доля верных на вопрос
по 12 ответам; состояние c3 — 3 сида обучения × 4 сэмпла. Δ — среднее по вопросам разности долей,
95 % — бутстрэп по вопросам."""
import json, re, os
import random
W = os.path.dirname(os.path.abspath(__file__))
checks = [l.rstrip("\n").split("\t") for l in open(f"{W}/checks.tsv", encoding="utf-8")
          if l.strip() and not l.startswith("#")][:-1]          # без вопроса об имени


def acc(paths):
    answers = [[] for _ in checks]
    for p in paths:
        for i, r in enumerate(json.load(open(f"{W}/{p}"))["rows"][:len(checks)]):
            assert r["question"] == checks[i][1]
            answers[i] += r["answers"]
    return [sum(bool(re.search(rx, a[:400], re.I)) for a in ans) / len(ans) for (rx, _), ans in zip(checks, answers)], len(answers[0])


arms = {"без промпта": ["gen_none_s12.json"], "промпт": ["gen_inline_s12.json"],
        "состояние c3": [f"gen_c3_fold_s{s}_s4.json" for s in (7, 8, 9)]}
A = {}
for k, ps in arms.items():
    A[k], n = acc(ps)
    print(f"{k:14} общие {sum(A[k]) / len(A[k]):.3f}  ({n} ответов на вопрос)")
for i, s in enumerate((7, 8, 9)):
    a, _ = acc([f"gen_c3_fold_s{s}_s4.json"])
    print(f"  сид {s}: {sum(a) / len(a):.3f}")
rng = random.Random(0)


def boot(x, y):
    d = [a - b for a, b in zip(x, y)]
    m = sum(d) / len(d)
    bs = sorted(sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(10000))
    return f"{m:+.3f} [{bs[250]:+.3f}; {bs[9750]:+.3f}]"


print(f"\nсостояние − промпт:       {boot(A['состояние c3'], A['промпт'])}")
print(f"состояние − без промпта:  {boot(A['состояние c3'], A['без промпта'])}")
print(f"промпт − без промпта:     {boot(A['промпт'], A['без промпта'])}")
