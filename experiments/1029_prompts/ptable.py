#!/usr/bin/env python3
"""Общая таблица пункта 3 (30.10): 5 промптов × 2 модели + Neuron из матрицы.
    python3 ptable.py > prompts_summary.md   (запускать из корня репозитория)"""
import json, os, re, sys
sys.path.insert(0, "experiments/1029_prompts"); sys.path.insert(0, "experiments/1027_matrix")
import numpy as np
import prompts, summ

W = "experiments/1029_prompts/p"; M = "experiments/1027_matrix/m"
kind_of = {q: k for k, _, q in prompts.train_lines()}
qs = [q for _, q in summ.checks]; rx = [x for x, _ in summ.checks]
S = summ.subsets()
models = ["g1d_0p4b_bf16", "g1d_1p5b_bf16"]
f = lambda a, b, idx: summ.fmt(summ.boot(a, b, idx, True), "{}")


def mat(paths):
    return np.stack([summ.per_q(p, qs, rx) for p in paths])


def comp(name, paths):
    if name not in prompts.P or name == "orion":
        return None
    return np.mean([[np.mean([bool(prompts.compliant(name, a)) for a in r["answers"]]) for r in json.load(open(p))["rows"][:len(qs)]] for p in paths])


print("# Пункт 3: разные системные промпты (29–30.10)\n")
print("g1d bf16, протокол c4 (fold, 300 шагов, 3 сида), один бинарник с матрицей (w1027/bin/gen-run). Общие — 199 вопросов "
      "(у english — 69 английских). Δ — к модели без промпта, {с×в} — двухуровневый бутстрэп 95 %. Соблюдение — доля ответов "
      "в нужной форме на 199 вопросах (у orion — «Орион» на 4 вопросах об имени, из 40). Факты учителя — сколько верных "
      "фактов (вид fact в исходных данных) прошло фильтр.\n")
print("| промпт | модель | факты учителя / всего взято | соблюдение: без / промпт / состояние | общие без | промпт Δ | состояние Δ | состояние Δ на фактах |")
print("|---|---|---|---|---|---|---|---|")
for name in ["neuron"] + list(prompts.P):
    for m in models:
        if name == "neuron":
            d = f"{M}/{m}"; inl = [f"{d}/v2_inline_s12.json"]; st = [f"{d}/v2_c4_s{s}_s4.json" for s in summ.SEEDS]
            tc = f"{d}/teacher.json"
        else:
            d = f"{W}/{name}/{m}"; inl = [f"{d}/v2_inline_s12.json"]; st = [f"{d}/v2_c4_s{s}_s4.json" for s in summ.SEEDS]
            tc = f"{d}/teacher.json"
        none = [f"{M}/{m}/v2_none_s12.json"]
        if not all(os.path.exists(p) for p in inl + st):
            print(f"| {name} | {m} | — |" + " |" * 5); continue
        idx = S["EN"] if name == "english" else S["все"]
        A, B, C = mat(none), mat(inl), mat(st)
        t = json.load(open(tc))
        kept = [x for x in t if x.get("kept", "1") == "1"]
        nf = sum(kind_of.get(x["question"]) == "fact" or x.get("kind") == "fact" and name == "neuron" for x in kept)
        if name == "neuron":
            nf = sum(x.get("kind") == "fact" for x in kept)
        else:
            nf = sum(kind_of.get(x["question"]) == "fact" for x in kept)
        if name == "orion":
            def nm(paths):
                n = 0; tot = 0
                for p in paths:
                    for r in json.load(open(p))["rows"][:4]:
                        n += sum(bool(re.search(r"орион|orion", a, re.I)) for a in r["answers"]); tot += len(r["answers"])
                return f"{n / tot:.2f}"
            cs = f"0 / {nm([f'{d}/id_inline_s10.json'])} / {nm([f'{d}/id_c4_s{s}_s10.json' for s in summ.SEEDS])}"
        elif name == "neuron":
            def nn(paths):
                n = 0; tot = 0
                for p in paths:
                    for r in json.load(open(p))["rows"][:4]:
                        n += sum(bool(re.search(r"neuron|нейрон", a, re.I)) for a in r["answers"]); tot += len(r["answers"])
                return f"{n / tot:.2f}"
            cs = f"0 / {nn([f'{d}/id_inline_s10.json'])} / {nn([f'{d}/id_c4_s{s}_s10.json' for s in summ.SEEDS])}"
        else:
            cs = f"{comp(name, none):.2f} / {comp(name, inl):.2f} / {comp(name, st):.2f}"
        print(f"| {name} | {m.split('_')[1]} | {nf} / {len(kept)} | {cs} | {A[:, idx].mean():.3f} | {f(B, A, idx)} | {f(C, A, idx)} | "
              + ("—" if name == "english" else f(C, A, S["факты"])) + " |")
