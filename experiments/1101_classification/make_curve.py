#!/usr/bin/env python3
"""Кривая по объёму данных: вложенные подмножества train.txt — первые 5 / 10 / 25 статей каждой рубрики (curve/train_n{25,50,125}.txt).
Запуск из этой папки после build.py."""
import collections, os, re
L = [l for l in open("train.txt", encoding="utf-8") if l.startswith("fact:")]
labs = [l.split("]*", 1)[1].split("[", 1)[0] for l in L]   # «fact: ^[\s*«\"]*рубрика[...» -> рубрика
os.makedirs("curve", exist_ok=True)
for n in (5, 10, 25):
    c = collections.Counter(); out = []
    for l, lb in zip(L, labs):
        if c[lb] < n: out.append(l); c[lb] += 1
    open(f"curve/train_n{n*5}.txt", "w", encoding="utf-8").write(
        f"# кривая по объёму данных — первые {n} статей каждой рубрики из train.txt (25 ⊂ 50 ⊂ 125 ⊂ 250)\n" + "".join(out))
    print(n * 5, dict(c))
