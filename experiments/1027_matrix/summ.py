#!/usr/bin/env python3
"""Сводка одной модели матрицы Neuron (27.10). python3 summ.py <папка модели> [<папка модели> ...]

Плечи: без промпта (none, 12 ответов на вопрос), промпт (inline, 12), состояние c4 и c (3 сида × 4 ответа).
Общие — checks.tsv из w1022/n15 (199 вопросов без имени), регэксп по первым 400 символам, как paired15.py.
Перенос — transfer.tsv (40 единиц × T/P), как transfer.py. Имя — questions.txt (10 ответов), как rates.py/admits.py.

Интервалы двух видов (95 %, 10 000 повторов):
  [в] — бутстрэп по вопросам, ответы сидов слиты (как в прежних сводках, сравнимо с черновиком);
  {с×в} — двухуровневый: сиды с возвращением, затем вопросы с возвращением (честнее при 3 сидах).
"""
import json, re, os, sys
import numpy as np

N15 = "data/neuron"
checks = [l.rstrip("\n").split("\t") for l in open(f"{N15}/checks.tsv", encoding="utf-8")
          if l.strip() and not l.startswith("#")][:-1]
NQ = len(checks)
trows = [l.rstrip("\n").split("\t") for l in open(f"{N15}/transfer.tsv", encoding="utf-8")
         if l.strip() and not l.startswith("#")]
SEEDS = (7, 8, 9)
B = 10000


def per_q(path, questions, regexes):
    rows = json.load(open(path))["rows"]
    out = []
    for i, (rx, q) in enumerate(zip(regexes, questions)):
        assert rows[i]["question"] == q, (path, i)
        a = rows[i]["answers"]
        out.append(sum(bool(re.search(rx, x[:400], re.I)) for x in a) / len(a))
    return np.array(out)


def arm_matrix(d, prefix, kind, questions, regexes):
    """Матрица [сиды × вопросы]; у none/inline — один «сид»."""
    if kind in ("none", "inline"):
        p = f"{d}/{prefix}_{kind}_s12.json"
        return per_q(p, questions, regexes)[None, :] if os.path.exists(p) else None
    ps = [f"{d}/{prefix}_{kind}_s{s}_s4.json" for s in SEEDS]
    if not all(os.path.exists(p) for p in ps):
        return None
    return np.stack([per_q(p, questions, regexes) for p in ps])


rng = np.random.default_rng(0)


def boot(a, b, idx, two_level):
    """a, b — матрицы [сиды × вопросы]; разность средних a − b на подмножестве idx."""
    idx = np.asarray(idx)
    a, b = a[:, idx], b[:, idx]
    m = a.mean() - b.mean()
    n = len(idx)
    qi = rng.integers(0, n, size=(B, n))
    if two_level:
        sa = rng.integers(0, a.shape[0], size=(B, a.shape[0]))
        sb = rng.integers(0, b.shape[0], size=(B, b.shape[0]))
        # среднее по выбранным сидам для каждого вопроса, затем по выбранным вопросам
        ma = np.stack([a[sa[k]].mean(0)[qi[k]].mean() for k in range(B)])
        mb = np.stack([b[sb[k]].mean(0)[qi[k]].mean() for k in range(B)])
        d = ma - mb
    else:
        d = (a.mean(0) - b.mean(0))[qi].mean(1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return m, lo, hi


def fmt(t, br="[]"):
    m, lo, hi = t
    return f"{m:+.3f} {br[0]}{lo:+.3f}; {hi:+.3f}{br[1]}"


def subsets():
    qs = [q for _, q in checks]
    ru = [bool(re.search("[а-яё]", q, re.I)) for q in qs]
    tr = [bool(re.search(r"Переведи|Translate", q)) for q in qs]
    ar = [bool(re.search(r"Сколько будет|What is \d|Чему равн|What is half|square root|squared", q)) for q in qs]
    r = range(NQ)
    return {"все": list(r), "перевод": [i for i in r if tr[i]], "арифметика": [i for i in r if ar[i]],
            "факты": [i for i in r if not tr[i] and not ar[i]],
            "RU": [i for i in r if ru[i]], "EN": [i for i in r if not ru[i]]}


def identity(d, tag):
    p = f"{d}/id_{tag}_s10.json"
    if not os.path.exists(p):
        return None
    yes = re.compile(r"^\W*(да|yes)\b|\bя\s*[—–-]?\s*chatgpt|(называюсь|являюсь)\s+chatgpt|\bi\s*am\s+chatgpt|\bi'm\s+chatgpt", re.I)
    no = re.compile(r"\bне\s+(являюсь\s+)?chatgpt|\bnot\s+chatgpt", re.I)
    out = {}
    for r in json.load(open(p))["rows"]:
        a = r["answers"]
        neu = sum(bool(re.search(r"neuron|нейрон", x, re.I)) for x in a)
        if "ChatGPT" in r["question"]:
            out["Ты ChatGPT? — признаёт"] = sum(bool(yes.search(x)) and not no.search(x) for x in a)
        out[r["question"]] = neu
    return out


def summarize(d):
    name = os.path.basename(d.rstrip("/"))
    L = [f"# {name}\n"]
    qs = [q for _, q in checks]; rx = [x for x, _ in checks]
    arms = {"без промпта": ("none",), "промпт": ("inline",), "состояние c4": ("c4",), "состояние c": ("c",)}
    G = {k: arm_matrix(d, "v2", v[0], qs, rx) for k, v in arms.items()}
    G = {k: v for k, v in G.items() if v is not None}
    S = subsets()
    L.append("## Общие (199 вопросов)\n")
    L.append("| плечо | " + " | ".join(f"{k} ({len(v)})" for k, v in S.items()) + " | сиды (все) |")
    L.append("|---|" + "---|" * (len(S) + 1))
    for k, m in G.items():
        seeds = " / ".join(f"{x:.3f}" for x in m.mean(1)) if m.shape[0] > 1 else ""
        L.append(f"| {k} | " + " | ".join(f"{m[:, idx].mean():.3f}" for idx in S.values()) + f" | {seeds} |")
    pairs = [("промпт", "без промпта"), ("состояние c4", "без промпта"), ("состояние c4", "промпт"),
             ("состояние c", "без промпта"), ("состояние c", "промпт"), ("состояние c", "состояние c4")]
    L.append("\n| разница | " + " | ".join(S) + " |")
    L.append("|---|" + "---|" * len(S))
    for a, b in pairs:
        if a in G and b in G:
            L.append(f"| {a} − {b} [в] | " + " | ".join(fmt(boot(G[a], G[b], idx, False)) for idx in S.values()) + " |")
            if G[a].shape[0] > 1 or G[b].shape[0] > 1:
                L.append(f"| {a} − {b} {{с×в}} | " + " | ".join(fmt(boot(G[a], G[b], idx, True), "{}") for idx in S.values()) + " |")
    # перенос
    tq = [r[3] for r in trows]; trx = [r[2] for r in trows]
    T = {k: arm_matrix(d, "v3", v[0], tq, trx) for k, v in arms.items()}
    T = {k: v for k, v in T.items() if v is not None}
    if "без промпта" in T:
        Ti = [i for i, r in enumerate(trows) if r[1] == "T"]; Pi = [i for i, r in enumerate(trows) if r[1] == "P"]
        L.append("\n## Перенос формы (40 единиц)\n")
        L.append("| плечо | T | P | − без промпта на T | на P |")
        L.append("|---|---|---|---|---|")
        for k, m in T.items():
            dT = fmt(boot(m, T["без промпта"], Ti, m.shape[0] > 1), "{}") if k != "без промпта" else ""
            dP = fmt(boot(m, T["без промпта"], Pi, m.shape[0] > 1), "{}") if k != "без промпта" else ""
            L.append(f"| {k} | {m[:, Ti].mean():.3f} | {m[:, Pi].mean():.3f} | {dT} | {dP} |")
    # имя
    ids = {"без промпта": [identity(d, "none")], "промпт": [identity(d, "inline")],
           "состояние c4": [identity(d, f"c4_s{s}") for s in SEEDS], "состояние c": [identity(d, f"c_s{s}") for s in SEEDS]}
    ids = {k: v for k, v in ids.items() if all(x is not None for x in v)}
    if ids:
        keys = list(next(iter(ids.values()))[0])
        L.append("\n## Личность (Neuron из 10; «признаёт» — признаёт себя ChatGPT)\n")
        L.append("| вопрос | " + " | ".join(ids) + " |")
        L.append("|---|" + "---|" * len(ids))
        for q in keys:
            L.append(f"| {q} | " + " | ".join(" / ".join(str(x[q]) for x in v) for v in ids.values()) + " |")
    # обучение
    L.append("\n## Обучение\n")
    for arm in [f"c4_s{s}" for s in SEEDS] + [f"c_s{s}" for s in SEEDS]:
        p = f"{d}/distill_{arm}.log"
        if os.path.exists(p):
            t = open(p, encoding="utf-8", errors="replace").read()
            last = re.findall(r"шаг 300: лосс [0-9.]+|лучший по отложенным[^)]*\)[^,)]*|сделано шагов \d+", t)
            pk = re.findall(r"(\d+)\s+peak memory footprint", t)
            L.append(f"- {arm}: {'; '.join(last[-2:]) if last else '—'}; пик процесса {int(pk[-1]) / 1e9:.2f} ГБ" if pk else f"- {arm}: —")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    for d in sys.argv[1:]:
        s = summarize(d)
        open(f"{d}/summary.md", "w", encoding="utf-8").write(s)
        print(s)
