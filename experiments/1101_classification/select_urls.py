#!/usr/bin/env python3
"""Список URL статей, отобранных build.py (train, test) и build_val.py (val), в том же порядке, что их файлы.
Повторяет отбор build.py по парам (текст, URL): random.shuffle даёт одну и ту же перестановку для списков одной длины.
Запуск из experiments/1101_classification рядом с final_lenta_new.csv.gz; выход — lenta_selection.tsv."""
import csv, gzip, random, re, sys
csv.field_size_limit(10**9)
TOP = {"Спорт": "спорт", "Экономика": "экономика", "Наука и техника": "наука", "Культура": "культура", "Путешествия": "путешествия"}
def clean(t): return re.sub(r"\s+", " ", t.replace(" ", " ")).strip()
by = {t: [] for t in TOP}
with gzip.open(sys.argv[1] if len(sys.argv) > 1 else "final_lenta_new.csv.gz", "rt", encoding="utf-8", newline="") as f:
    for r in csv.DictReader(f):
        if r["topic"] in TOP and " :: " not in r["news"]:
            by[r["topic"]].append((clean(r["news"]), r["url"]))
rng = random.Random(1101); test, train, val = [], [], []
for t, lab in TOP.items():
    docs = by[t]; rng.shuffle(docs)
    longd = [d for d in docs if len(d[0]) >= 3000][:30]
    ls = {d[0] for d in longd}
    rest = [d for d in docs if d[0] not in ls and len(d[0]) >= 400]
    test += [(lab, d[1]) for d in longd]; train += [(lab, d[1]) for d in rest[:50]]; val += [(lab, d[1]) for d in rest[50:70]]
rng.shuffle(test); rng.shuffle(train)
rng2 = random.Random(1101)   # build_val.py: те же перестановки по рубрикам, затем перемешивание val
for t in TOP: rng2.shuffle(list(range(len(by[t]))))
rng2.shuffle(val)
with open("lenta_selection.tsv", "w", encoding="utf-8") as f:
    f.write("# раздел\tномер\tрубрика\tURL — порядок совпадает с train.txt, test*.tsv (одна статья на три длины), val.tsv\n")
    for name, L in (("train", train), ("test", test), ("val", val)):
        for i, (lab, url) in enumerate(L): f.write(f"{name}\t{i}\t{lab}\t{url}\n")
print(len(train), len(test), len(val))
