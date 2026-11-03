#!/usr/bin/env python3
"""Промахи регэкспов checks.tsv по ответам плеча: вопрос, доля верных и разные непойманные ответы
(первые 110 символов) — для поиска ложных промахов (верный ответ другими словами) и ложных попаданий.
    python3 misses.py arm.json [--hits]   (--hits — показать и пойманные, для ложных попаданий)"""
import json, re, sys, os
W = os.path.dirname(os.path.abspath(__file__))
checks = [l.rstrip("\n").split("\t") for l in open(f"{W}/checks.tsv", encoding="utf-8")
          if l.strip() and not l.startswith("#")]
rows = json.load(open(sys.argv[1]))["rows"]
for i, ((rx, q), r) in enumerate(zip(checks, rows)):
    assert r["question"] == q
    hit = [bool(re.search(rx, a[:400], re.I)) for a in r["answers"]]
    show = [a for a, h in zip(r["answers"], hit) if not h or "--hits" in sys.argv]
    print(f"\n[{i}] {q}  /{rx}/  {sum(hit)}/{len(hit)}")
    for a in dict.fromkeys(x[:110].replace("\n", " ⏎ ") for x in show):
        print("   ", a)
