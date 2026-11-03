#!/usr/bin/env python3
"""30.10, пункт 2: валидация для выбора LR контроля — факты из train_questions4, на которые учитель (та же модель
с промптом Neuron) не дал ни одного верного ответа: в обучение они не попали, с checks.tsv не пересекаются.
    python3 ctl_val.py build <teacher.json> <out_dir>      -> val.tsv (регэксп \t вопрос), val_q.txt
    python3 ctl_val.py score <val.tsv> <answers.json>      -> доля верных"""
import json, re, sys
N15 = "data/neuron/train_questions4.txt"
if sys.argv[1] == "build":
    facts = {}
    for l in open(N15, encoding="utf-8"):
        if l.startswith("fact:"):
            rx, q = l[5:].strip().split(" :: "); facts[q] = rx
    t = json.load(open(sys.argv[2]))
    kept = {}
    for x in t:
        if x["question"] in facts:
            kept[x["question"]] = kept.get(x["question"], 0) + (x.get("kept", "1") == "1")
    val = [(facts[q], q) for q in facts if kept.get(q, 0) == 0]
    open(f"{sys.argv[3]}/val.tsv", "w", encoding="utf-8").write("".join(f"{r}\t{q}\n" for r, q in val))
    open(f"{sys.argv[3]}/val_q.txt", "w", encoding="utf-8").write("".join(f"{q}\n" for _, q in val))
    print(f"валидация: {len(val)} фактов (из {len(facts)}; у учителя 0 верных)")
else:
    val = [l.rstrip("\n").split("\t") for l in open(sys.argv[2], encoding="utf-8")]
    rows = json.load(open(sys.argv[3]))["rows"]
    acc = [sum(bool(re.search(rx, a[:400], re.I)) for a in r["answers"]) / len(r["answers"]) for (rx, q), r in zip(val, rows)]
    assert all(r["question"] == q for (rx, q), r in zip(val, rows))
    print(f"{sum(acc) / len(acc):.4f}")
