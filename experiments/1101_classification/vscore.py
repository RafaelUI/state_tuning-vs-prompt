#!/usr/bin/env python3
"""01.11: строгая точность на валидации (val.tsv) — для выбора LR сильного контроля. python3 vscore.py <ответы.json>"""
import json, re, sys
LAB = ["спорт", "экономика", "наука", "культура", "путешествия"]
rx = re.compile(r"^[\s*«\"]*(" + "|".join(LAB) + r")[\s*»\".!]*$", re.I)
gold = [l.split("\t")[0] for l in open("experiments/1101_classification/val_labels.tsv", encoding="utf-8")]
R = json.load(open(sys.argv[1]))["rows"]; assert len(R) == len(gold)
ok = [bool((m := rx.match(a.strip()))) and m.group(1).lower() == g for r, g in zip(R, gold) for a in r["answers"]]
print(f"{sum(ok) / len(ok):.4f}")
