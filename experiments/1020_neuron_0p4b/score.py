#!/usr/bin/env python3
"""Доля верных ответов на checks.tsv (регэксп по первым 400 символам ответа, без учёта регистра).
    python3 score.py arm1.json arm2.json …   — последняя строка checks.tsv (имя) считается отдельно."""
import json, re, sys, os
W = os.path.dirname(os.path.abspath(__file__))
checks = [l.rstrip("\n").split("\t") for l in open(f"{W}/checks.tsv", encoding="utf-8")
          if l.strip() and not l.startswith("#")]
res = {}
for p in sys.argv[1:]:
    rows = json.load(open(p))["rows"]
    assert [r["question"] for r in rows] == [q for _, q in checks], p
    per = [sum(bool(re.search(rx, a[:400], re.I)) for a in r["answers"]) / len(r["answers"])
           for (rx, _), r in zip(checks, rows)]
    res[p] = per
    gen = per[:-1]
    print(f"{os.path.basename(p):34} общие {sum(gen)/len(gen):.3f} ({len(gen)} вопр.)  имя {per[-1]:.2f}")
if len(res) > 1:
    names = list(res)
    print("  вопросы, где плечи расходятся на 0.5 и больше: " + " | ".join(os.path.basename(n) for n in names))
    for i, (rx, q) in enumerate(checks[:-1]):
        vals = [res[n][i] for n in names]
        if max(vals) - min(vals) >= 0.5:
            print(f"    {q[:44]:44} " + "  ".join(f"{v:.2f}" for v in vals))
