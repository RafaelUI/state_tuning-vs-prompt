#!/usr/bin/env python3
"""31.10: кэши учителя для двух плеч c4p (решение Алексея 31.10: оба плеча).
  teacher_d.json — отобранный teacher.json без новых 32 вопросов: учитель догенерирует их с --teacher-samples 16 (сиды 7..22;
                   первые 4 совпадают с прежними), потом ручная сверка (curate.py drop).
  teacher_h.json — отобранный teacher.json, новые 32 вопроса заменены ручными ответами hand_answers.tsv (по 2 на вопрос)."""
import json, sys
new = [l.split(" :: ")[1].strip() for l in open("experiments/1031_prov_training/train_prov.txt", encoding="utf-8") if l.startswith("fact: ^(?!")]
hand = [l.rstrip("\n").split("\t") for l in open("experiments/1031_prov_training/hand_answers.tsv", encoding="utf-8") if l.strip() and not l.startswith("#")]
assert sorted(set(q for q, _ in hand)) == sorted(new), "вопросы hand_answers.tsv != новые вопросы train_prov.txt"
for m in sys.argv[1:]:
    c = json.load(open(f"experiments/1031_prov_training/pt/{m}/teacher.json"))
    old = [d for d in c if d["question"] not in new]
    json.dump(old, open(f"experiments/1031_prov_training/pt/{m}/teacher_d.json", "w"), ensure_ascii=False, indent=2, sort_keys=True)
    h = old + [{"kind": "fact", "question": q, "answer": a, "raw": "\n" + a, "kept": "1"} for q, a in hand]
    json.dump(h, open(f"experiments/1031_prov_training/pt/{m}/teacher_h.json", "w"), ensure_ascii=False, indent=2, sort_keys=True)
    print(m, "старых", len(old), "ручных", len(hand))
