#!/usr/bin/env python3
"""01.11: сводка классификации (run_cls.sh; плечи без промпта / промпт / состояние / LoRA / top4, все на золотых метках). venv rwkv-metal: python clssum.py > cls_summary.md
Строго: ответ — ровно одно слово-рубрика из списка (кавычки/точка допустимы) и оно верное. Мягко: первая рубрика из списка,
встретившаяся в ответе (в любом виде), верная. Соблюдение формата: ответ — ровно одна рубрика (верная или нет).
Разности — двухуровневый бутстрэп {с×в} (сиды × статьи, 95 %, 10 000 повторов); статьи у трёх длин одни и те же."""
import json, os, re
import numpy as np
C = "experiments/1101_classification"; O = f"{C}/g1d_0p4b_bf16"
LAB = ["спорт", "экономика", "наука", "культура", "путешествия"]
STRICT = re.compile(r"^[\s*«\"]*(" + "|".join(LAB) + r")[\s*»\".!]*$", re.I)
ANY = re.compile(r"(" + "|".join(l[:-1] if l != "наука" else "наук" for l in LAB) + r")", re.I)   # основы: спор|экономик|наук|культур|путешестви
STEM = {"спор": "спорт", "экономик": "экономика", "наук": "наука", "культур": "культура", "путешестви": "путешествия"}
rows = [l.rstrip("\n").split("\t") for l in open(f"{C}/test_labels.tsv", encoding="utf-8") if not l.startswith("#")]
GOLD = {k: [r[0] for r in rows if r[1] == k] for k in ("short", "mid", "long")}
def score(a, g):
    m = STRICT.match(a.strip()); comp = bool(m)
    strict = comp and m.group(1).lower() == g
    f = ANY.search(a); soft = bool(f) and STEM[f.group(1).lower()[:len(f.group(1))].lower()] == g if f else False
    return strict, soft, comp
def mats(arm, k):
    ps = [f"{O}/{k}_{arm}.json"] if arm in ("none", "inline") else [f"{O}/{k}_{arm}_s{s}.json" for s in (7, 8, 9)]
    if not all(os.path.exists(p) for p in ps): return None
    out = []
    for p in ps:
        R = json.load(open(p))["rows"]; assert len(R) == len(GOLD[k])
        out.append(np.array([[np.mean([score(a, g)[i] for a in r["answers"]]) for r, g in zip(R, GOLD[k])] for i in range(3)]))
    return np.stack(out)          # [сиды × 3 метрики × статьи]
rng = np.random.default_rng(0)
def boot(a, b, B=10000):
    m = a.mean() - b.mean(); n = a.shape[1]; d = np.empty(B)
    for i in range(B):
        q = rng.integers(0, n, n); sa = rng.integers(0, a.shape[0], a.shape[0]); sb = rng.integers(0, b.shape[0], b.shape[0])
        d[i] = a[sa].mean(0)[q].mean() - b[sb].mean(0)[q].mean()
    lo, hi = np.percentile(d, [2.5, 97.5]); return f"{m:+.3f} {{{lo:+.3f}; {hi:+.3f}}}"
NM = {"none": "без промпта", "inline": "промпт", "cg": "состояние (золотые метки)", "lora": "LoRA r16", "top": "верхние 4 слоя"}
L = ("short", "mid", "long")
M = {a: {k: mats(a, k) for k in L} for a in NM}
print("# Классификация новостей, g1d 0.4B (01.11)\n\n" + __doc__.split("\n", 1)[1] + "\n")
for i, title in enumerate(("Точность, строго", "Точность, мягко", "Соблюдение формата")):
    print(f"## {title}\n\n| плечо | ~100 ток. | ~400 | ~1000 | сиды (~1000) |\n|---|---|---|---|---|")
    for a, nm in NM.items():
        cells = [f"{M[a][k][:, i].mean():.3f}" if M[a][k] is not None else "—" for k in L]
        sd = " / ".join(f"{x:.3f}" for x in M[a]["long"][:, i].mean(1)) if M[a]["long"] is not None and M[a]["long"].shape[0] > 1 else ""
        print(f"| {nm} | " + " | ".join(cells) + f" | {sd} |")
    print()
print("## Разности {с×в}, строгая точность\n\n| разность | ~100 | ~400 | ~1000 |\n|---|---|---|---|")
for a, b in (("cg", "inline"), ("cg", "lora"), ("cg", "top"), ("lora", "top"), ("inline", "none")):
    if all(M[x][k] is not None for x in (a, b) for k in L):
        print(f"| {NM[a]} − {NM[b]} | " + " | ".join(boot(M[a][k][:, 0], M[b][k][:, 0]) for k in L) + " |")
print("\n| длина: ~1000 − ~100 | строго |\n|---|---|")
for a in NM:
    if M[a]["long"] is not None and M[a]["short"] is not None:
        print(f"| {NM[a]} | {boot(M[a]['long'][:, 0], M[a]['short'][:, 0])} |")
