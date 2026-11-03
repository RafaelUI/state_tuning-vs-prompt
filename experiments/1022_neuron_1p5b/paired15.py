#!/usr/bin/env python3
"""Парное сравнение общих способностей на расширенном checks.tsv (21.10: 199 вопросов без имени).
Доля верных на вопрос по 12 ответам (без промпта, промпт — 12 сэмплов; состояние — 3 сида обучения × 4
сэмпла). Δ — среднее по вопросам разности долей, 95 % — бутстрэп по вопросам (10 000). Срезы: прежние 39
вопросов / добавленные 160, русские / английские. Плечи без файлов пропускаются."""
import json, re, os, random
W = os.path.dirname(os.path.abspath(__file__))
checks = [l.rstrip("\n").split("\t") for l in open(f"{W}/checks.tsv", encoding="utf-8")
          if l.strip() and not l.startswith("#")][:-1]          # без вопроса об имени
N = len(checks)


def acc(paths):
    answers = [[] for _ in checks]
    for p in paths:
        for i, r in enumerate(json.load(open(f"{W}/{p}"))["rows"][:N]):
            assert r["question"] == checks[i][1], (p, i)
            answers[i] += r["answers"]
    return [sum(bool(re.search(rx, a[:400], re.I)) for a in ans) / len(ans) for (rx, _), ans in zip(checks, answers)], len(answers[0])


arms = {"без промпта": ["v2_none_s12.json"], "промпт": ["v2_inline_s12.json"], "состояние 1.5B": [f"v2_s15_fold_s{s}_s4.json" for s in (7, 8, 9)],
        "состояние c3": [f"v2_c3_fold_s{s}_s4.json" for s in (7, 8, 9)],
        "состояние c4": [f"v2_c4_fold_s{s}_s4.json" for s in (7, 8, 9)]}
A = {}
for k, ps in arms.items():
    if not all(os.path.exists(f"{W}/{p}") for p in ps):
        print(f"{k:14} — нет всех файлов"); continue
    A[k], n = acc(ps)
    seeds = "" if len(ps) == 1 else "  сиды: " + " / ".join(f"{sum(acc([p])[0]) / N:.3f}" for p in ps)
    print(f"{k:14} общие {sum(A[k]) / N:.3f}  ({n} ответов на вопрос){seeds}")
ru = [bool(re.search("[а-яё]", q, re.I)) for _, q in checks]
subsets = {"все": list(range(N)), "прежние 39": list(range(39)), "новые 160": list(range(39, N)),
           "русские": [i for i in range(N) if ru[i]], "английские": [i for i in range(N) if not ru[i]]}
# вид вопроса: перевод и арифметика — шаблоны, которые есть и в обучающих фактах; остальное — знания
tr = [bool(re.search(r"Переведи|Translate", q)) for _, q in checks]
ar = [bool(re.search(r"Сколько будет|What is \d|Чему равн|What is half|square root|squared", q)) for _, q in checks]
subsets["перевод"] = [i for i in range(N) if tr[i]]
subsets["арифметика"] = [i for i in range(N) if ar[i]]
subsets["факты"] = [i for i in range(N) if not tr[i] and not ar[i]]
rng = random.Random(0)


def boot(x, y, idx):
    d = [x[i] - y[i] for i in idx]
    m = sum(d) / len(d)
    bs = sorted(sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(10000))
    return f"{m:+.3f} [{bs[250]:+.3f}; {bs[9750]:+.3f}]"


pairs = [("промпт", "без промпта"), ("состояние 1.5B", "без промпта"), ("состояние 1.5B", "промпт")]
print("\n| разница | " + " | ".join(f"{k} (n={len(v)})" for k, v in subsets.items()) + " |")
print("|---|" + "---|" * len(subsets))
for a, b in pairs:
    if a in A and b in A:
        print(f"| {a} − {b} | " + " | ".join(boot(A[a], A[b], idx) for idx in subsets.values()) + " |")
print("\n| плечо | " + " | ".join(subsets) + " |\n|---|" + "---|" * len(subsets))
for k in A:
    print(f"| {k} | " + " | ".join(f"{sum(A[k][i] for i in idx) / len(idx):.3f}" for idx in subsets.values()) + " |")
