#!/usr/bin/env python3
"""Как misses.py, но по журналу gen-run (ответы, ещё не записанные в JSON): «[i] вопрос» и «  #k: ответ»."""
import re, sys, os
W = os.path.dirname(os.path.abspath(__file__))
checks = [l.rstrip("\n").split("\t") for l in open(f"{W}/checks.tsv", encoding="utf-8") if l.strip() and not l.startswith("#")]
rows, cur = [], None
for l in open(sys.argv[1], encoding="utf-8"):
    m = re.match(r"^\[(\d+)\] (.*)$", l.rstrip("\n"))
    if m: cur = []; rows.append((m.group(2), cur)); continue
    m = re.match(r"^  #\d+: (.*)$", l.rstrip("\n"))
    if m and cur is not None: cur.append(m.group(1).replace(" ⏎ ", "\n"))
lo = int(sys.argv[2]) if len(sys.argv) > 2 else 0
for i, ((rx, q), (q2, ans)) in enumerate(zip(checks, rows)):
    assert q == q2, (q, q2)
    if i < lo or len(ans) == 0: continue
    hit = [bool(re.search(rx, a[:400], re.I)) for a in ans]
    miss = [a for a, h in zip(ans, hit) if not h]
    print(f"\n[{i}] {q} /{rx}/ {sum(hit)}/{len(hit)}")
    for a in list(dict.fromkeys(x[:100].replace("\n", " ⏎ ") for x in miss))[:6]: print("   -", a)
