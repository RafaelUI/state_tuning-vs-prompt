#!/usr/bin/env python3
"""Доли ответов по проверочным вопросам: «Neuron», чужое имя, упоминание стихов.
    python3 rates.py g04_none_s10.json g04_inline_s10.json …"""
import json, re, sys
for p in sys.argv[1:]:
    print(f"\n{p}")
    for r in json.load(open(p))["rows"]:
        a = r["answers"]; n = len(a)
        neu = sum(bool(re.search(r"neuron|нейрон", x, re.I)) for x in a)
        oth = sum(bool(re.search(r"chatgpt|openai|qwen|alibaba|\bgpt|алекс|alex", x, re.I)) for x in a)
        poem = sum(bool(re.search(r"стих|poem|poet|поэ", x, re.I)) for x in a)
        print(f"  {r['question'][:40]:40} Neuron {neu:2}/{n}  чужое имя {oth:2}/{n}  стихи {poem:2}/{n}")
