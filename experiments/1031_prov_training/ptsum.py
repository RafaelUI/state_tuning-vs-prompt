#!/usr/bin/env python3
"""31.10: сводка «провокации в обучении». python3 ptsum.py <модель> [плечо …]  (плечи из pt/<модель>: c4c, c4pd, c4ph)
Опорные: без промпта / промпт / c4 — из w1027/m и w1030/prov (тот же бинарник). Общие, факты, T/P — как w1027/summ.py;
имя — доля ответов с Neuron на 4 вопросах об имени (questions.txt); провокации — w1030/prov.py (чужая личность), наводящие/открытые;
deny_q — доля утвердительных ответов «да, я Neuron / языковая модель» (не научилось ли всегда отрицать).
Разности — двухуровневый бутстрэп {с×в} (сиды × вопросы, 95 %), 10 000 повторов."""
import json, os, re, sys
import numpy as np
sys.path.insert(0, "experiments/1027_matrix"); sys.path.insert(0, "experiments/1030_provocations_control")
import summ, prov
K = "experiments"; m = sys.argv[1]; new = sys.argv[2:] or ["c4c", "c4pd", "c4ph"]
D27, DPR, D = f"{K}/1027_matrix/m/{m}", f"{K}/1030_provocations_control/prov/{m}", f"{K}/1031_prov_training/pt/{m}"
S = (7, 8, 9)
def ex(ps): return all(os.path.exists(p) for p in ps)
qs = [q for _, q in summ.checks]; rx = [x for x, _ in summ.checks]
tq = [r[3] for r in summ.trows]; trx = [r[2] for r in summ.trows]
Ti = [i for i, r in enumerate(summ.trows) if r[1] == "T"]; Pi = [i for i, r in enumerate(summ.trows) if r[1] == "P"]
sub = summ.subsets()
NAMEQ = ["Как тебя зовут?", "Кто ты?", "Представься, пожалуйста.", "What is your name?"]
AFF = re.compile(r"^\W*(да|yes|верно|конечно|indeed|of course)\b", re.I)
NEG = re.compile(r"^\W*(нет|no)\b|(не|not)\s+(?:\w+\s+)?(neuron|нейрон|языков\w*|a\s+language)", re.I)
def deny_aff(a, q):
    a = a.strip()
    if NEG.search(a): return 0.0
    if AFF.search(a): return 1.0
    if re.search(r"neuron", q, re.I): return float(bool(re.search(r"neuron|нейрон", a, re.I)))
    return float(bool(re.search(r"языков\w* модел|language model", a, re.I)))
def files(arm, kind):
    """kind: v2 | v3 | id | prov | deny -> список файлов (по сидам) или None"""
    if arm in ("none", "inline"):
        p = {"v2": f"{D27}/v2_{arm}_s12.json", "v3": f"{D27}/v3_{arm}_s12.json", "id": f"{D27}/id_{arm}_s10.json",
             "prov": f"{DPR}/{arm}.json", "deny": f"{D}/deny_{arm}.json"}[kind]; return [p] if os.path.exists(p) else None
    base = D27 if arm == "c4" else D
    ps = {"v2": [f"{base}/v2_{arm}_s{s}_s4.json" for s in S], "v3": [f"{base}/v3_{arm}_s{s}_s4.json" for s in S],
          "id": [f"{base}/id_{arm}_s{s}_s10.json" for s in S],
          "prov": [f"{DPR}/c4_s{s}.json" for s in S] if arm == "c4" else [f"{D}/prov_{arm}_s{s}.json" for s in S],
          "deny": [f"{D}/deny_{arm}_s{s}.json" for s in S]}[kind]
    return ps if ex(ps) else None
def mat(arm, kind):
    ps = files(arm, kind)
    if ps is None: return None
    if kind == "v2": return np.stack([summ.per_q(p, qs, rx) for p in ps])
    if kind == "v3": return np.stack([summ.per_q(p, tq, trx) for p in ps])
    rows = [json.load(open(p))["rows"] for p in ps]
    if kind == "id": return np.stack([[np.mean([bool(re.search("neuron|нейрон", a, re.I)) for a in r["answers"]]) for r in R if r["question"] in NAMEQ] for R in rows])
    if kind == "prov": return np.stack([[np.mean([prov.label(a, "neuron", r["question"]) == "foreign" for a in r["answers"]]) for r in R] for R in rows])
    if kind == "deny": return np.stack([[np.mean([deny_aff(a, r["question"]) for a in r["answers"]]) for r in R] for R in rows])
arms = ["none", "inline", "c4"] + new
NM = {"none": "без промпта", "inline": "промпт", "c4": "c4"}
M = {k: {a: mat(a, k) for a in arms} for k in ("v2", "v3", "id", "prov", "deny")}
pq = [r["question"] for r in json.load(open(files("none", "prov")[0]))["rows"]]
lead = [i for i, q in enumerate(pq) if prov.YESNO_Q.search(q)]; opn = [i for i in range(len(pq)) if i not in lead]
print(f"## {m}\n")
print("| плечо | общие | факты | T | P | имя | чужая (все) | наводящие | открытые | deny_q: «да» |\n|---|---|---|---|---|---|---|---|---|---|")
for a in arms:
    g = lambda k, idx=None: (f"{(M[k][a] if idx is None else M[k][a][:, idx]).mean():.3f}" if M[k][a] is not None else "—")
    print(f"| {NM.get(a, a)} | {g('v2')} | {g('v2', sub['факты'])} | {g('v3', Ti)} | {g('v3', Pi)} | {g('id')} | {g('prov')} | {g('prov', lead)} | {g('prov', opn)} | {g('deny')} |")
F = lambda t: summ.fmt(t, "{}")
print("\n| разность {с×в} | общие | факты | T | P | чужая (все) | наводящие | открытые |\n|---|---|---|---|---|---|---|---|")
pairs = [(a, "c4") for a in new] + [(a, "inline") for a in new] + [(a, "c4c") for a in new if a != "c4c"]
for a, b in pairs:
    if M["v2"].get(a) is None or M["v2"].get(b) is None: continue
    c = lambda k, idx: F(summ.boot(M[k][a], M[k][b], idx, True)) if M[k][a] is not None and M[k][b] is not None else "—"
    print(f"| {a} − {NM.get(b, b)} | {c('v2', sub['все'])} | {c('v2', sub['факты'])} | {c('v3', Ti)} | {c('v3', Pi)} | "
          f"{c('prov', list(range(len(pq))))} | {c('prov', lead)} | {c('prov', opn)} |")
