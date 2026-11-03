#!/usr/bin/env python3
"""01.11: валидация для подбора LR сильного контроля (LoRA / top4) — 20 статей на рубрику, не из обучения и не из проверки
(тот же порядок перемешивания, что build.py; берутся следующие после обучающих), длина по кругу ~100/~400/~1000 токенов.
Выход: val_q.txt, val.tsv (метка \t текст)."""
import csv, gzip, random, re, sys
sys.path.insert(0, ".")   # запуск из experiments/1101_classification
import importlib.util
spec = importlib.util.spec_from_file_location("b", "build.py")
src = open(spec.origin, encoding="utf-8").read().split("rng = random.Random(1101)")[0]   # константы, clean, cut, чтение корпуса
exec(src)
rng = random.Random(1101); val = []
test_docs = {l.rstrip("\n").split("\t")[2] for l in open("test.tsv", encoding="utf-8") if not l.startswith("#")}
train_docs = {l.split(" :: ", 1)[1].rstrip("\n") for l in open("train.txt", encoding="utf-8") if l.startswith("fact:")}
for t, lab in TOP.items():
    docs = by[t]; rng.shuffle(docs)                                    # тот же порядок, что в build.py
    longd = [d for d in docs if len(d) >= 3000][:30]
    rest = [d for d in docs if d not in set(longd) and len(d) >= 400]
    for i, d in enumerate(rest[50:70]):
        v = cut(d, list(TARGET.values())[i % 3]); assert v not in train_docs and v not in test_docs
        val.append((lab, v))
rng.shuffle(val)
open("val_q.txt", "w", encoding="utf-8").write("".join(v + "\n" for _, v in val))
open("val.tsv", "w", encoding="utf-8").write("".join(f"{l}\t{v}\n" for l, v in val))
print(len(val), "статей валидации")
