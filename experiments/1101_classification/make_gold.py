#!/usr/bin/env python3
"""Кэш «учителя» для обучения на золотых метках: ответ — сама рубрика (gold_answers.tsv из build.py). Запуск из этой папки."""
import json
g = [l.rstrip("\n").split("\t") for l in open("gold_answers.tsv", encoding="utf-8")]
json.dump([{"kind": "fact", "question": d, "answer": lab, "raw": "\n" + lab, "kept": "1"} for d, lab in g],
          open("teacher_g.json", "w"), ensure_ascii=False, indent=1, sort_keys=True)
print(len(g))
