#!/usr/bin/env python3
"""Пункт 3 (29.10): разные системные промпты. Без правки кода gen-run — фильтр учителя через вид «fact»
(«fact: регэксп :: вопрос»: берётся только не оборванный ответ, совпавший с регэкспом). Для каждого промпта
регэксп = соблюдение промпта И (для фактов) верный ответ.

    python3 prompts.py build <папка>        — <папка>/<промпт>/{system.txt, train.txt}
    python3 prompts.py score <папка> <промпт> <модель>  — соблюдение + общие (none — из матрицы w1027/m)
"""
import json, os, re, sys

N15 = "data/neuron"
M = "experiments/1027_matrix/m"

# имя -> (системная строка, отброс пересказа промпта, регэксп соблюдения для учителя)
SENT = r"[^.!?\n]+[.!?…]?\s*$"
NOLIST = r"(?![\s\S]*(?:^|\n)\s*(?:[-•]|\d+[.)])\s)[^*#]+$"
P = {
    "orion": ("Ты — Орион, языковая модель.", "Ты — Орион|System:", None),
    "plain": ("Пиши обычным текстом, без markdown: без звёздочек, решёток и списков.", "markdown|System:", NOLIST),
    "short": ("Отвечай кратко, одним предложением.", "одним предложением|System:", SENT),
    "english": ("Always answer in English.", "answer in English|System:", r"[^А-Яа-яЁё]+$"),
    "json": ('Отвечай только в формате JSON: {"answer": "…"}.', "формате JSON|System:", r'\s*\{[\s\S]*"answer"[\s\S]*\}\s*$'),
}


def train_lines():
    out = []
    for l in open(f"{N15}/train_questions4.txt", encoding="utf-8"):
        l = l.rstrip("\n")
        if not l.strip() or l.startswith("#"):
            continue
        kind, rest = l.split(":", 1)
        rest = rest.strip()
        if kind == "fact":
            rx, q = rest.split(" :: ")
            out.append((kind, rx, q))
        else:
            out.append((kind, None, rest))
    return out


def build(root):
    for name, (system, reject, comp) in P.items():
        d = f"{root}/{name}"; os.makedirs(d, exist_ok=True)
        open(f"{d}/system.txt", "w", encoding="utf-8").write(system)
        open(f"{d}/reject.txt", "w", encoding="utf-8").write(reject)
        L = [f"# 29.10, промпт «{name}»: {system}",
             "# всё через вид fact: регэксп = соблюдение промпта И (у фактов) верный ответ; с checks.tsv не пересекается"]
        n = {"взято": 0, "пропущено": 0}
        for kind, rx, q in train_lines():
            if name == "orion":
                r = {"id": r"орион|orion", "poem": r"\n[\s\S]*\n", "fact": rx}.get(kind, r"[\s\S]")
            else:
                if name == "english" and kind == "fact" and not re.search(r"[A-Za-z]", rx):
                    n["пропущено"] += 1; continue        # верный ответ — русский: с «отвечай по-английски» несовместим
                r = f"^(?=[\\s\\S]*?(?:{rx})){comp}" if kind == "fact" else f"^{comp}"
            L.append(f"fact: {r} :: {q}")
            n["взято"] += 1
        open(f"{d}/train.txt", "w", encoding="utf-8").write("\n".join(L) + "\n")
        print(name, n, system)


# ---- оценка ----
def compliant(name, a):
    a = a.strip()
    if name == "plain":
        return not re.search(r"[*#]", a) and not re.search(r"(?:^|\n)\s*(?:[-•]|\d+[.)])\s", a)
    if name == "short":
        return "\n" not in a and len([s for s in re.split(r"(?<=[.!?…])\s+", a) if s.strip()]) == 1
    if name == "english":
        return not re.search(r"[А-Яа-яЁё]", a)
    if name == "json":
        try:
            j = json.loads(a)
            return isinstance(j, dict) and "answer" in j
        except Exception:
            return False
    return None


def score(root, name, model):
    sys.path.insert(0, "experiments/1027_matrix")
    import summ
    import numpy as np
    d = f"{root}/{name}/{model}"
    qs = [q for _, q in summ.checks]; rx = [x for x, _ in summ.checks]
    arms = {"без промпта": [f"{M}/{model}/v2_none_s12.json"], "промпт": [f"{d}/v2_inline_s12.json"],
            "состояние": [f"{d}/v2_c4_s{s}_s4.json" for s in summ.SEEDS]}
    arms = {k: v for k, v in arms.items() if all(os.path.exists(p) for p in v)}
    ru = [bool(re.search("[а-яё]", q, re.I)) for q in qs]
    S = summ.subsets()
    L = [f"# {name} ({open(f'{root}/{name}/system.txt', encoding='utf-8').read()}) — {model}\n"]
    G = {k: np.stack([summ.per_q(p, qs, rx) for p in v]) for k, v in arms.items()}
    C = {}
    for k, v in arms.items():
        mats = []
        for p in v:
            rows = json.load(open(p))["rows"][:len(qs)]
            mats.append([np.mean([bool(compliant(name, a)) for a in r["answers"]]) for r in rows] if name != "orion" else [0] * len(qs))
        C[k] = np.array(mats)
    if name != "orion":
        L.append("## Соблюдение промпта (доля ответов на 199 вопросах)\n")
        L.append("| плечо | все | RU | EN | сиды |\n|---|---|---|---|---|")
        for k, m in C.items():
            seeds = " / ".join(f"{x:.3f}" for x in m.mean(1)) if m.shape[0] > 1 else ""
            L.append(f"| {k} | {m.mean():.3f} | {m[:, [i for i in range(len(qs)) if ru[i]]].mean():.3f} | "
                     f"{m[:, [i for i in range(len(qs)) if not ru[i]]].mean():.3f} | {seeds} |")
        if "состояние" in C and "промпт" in C:
            L.append(f"\nсостояние − промпт {{с×в}}: {summ.fmt(summ.boot(C['состояние'], C['промпт'], S['все'], True), '{}')}")
    sub = {"все": S["все"], "факты": S["факты"], "RU": S["RU"], "EN": S["EN"]} if name != "english" else {"EN (69)": S["EN"]}
    L.append("\n## Общие (верный ответ" + (", только английские вопросы: русские регэкспы ответ по-английски не засчитывают" if name == "english" else "") + ")\n")
    L.append("| плечо | " + " | ".join(sub) + " |\n|---|" + "---|" * len(sub))
    for k, m in G.items():
        L.append(f"| {k} | " + " | ".join(f"{m[:, idx].mean():.3f}" for idx in sub.values()) + " |")
    L.append("\n| разница {с×в} | " + " | ".join(sub) + " |\n|---|" + "---|" * len(sub))
    for a, b in (("промпт", "без промпта"), ("состояние", "без промпта"), ("состояние", "промпт")):
        if a in G and b in G:
            L.append(f"| {a} − {b} | " + " | ".join(summ.fmt(summ.boot(G[a], G[b], idx, True), "{}") for idx in sub.values()) + " |")
    if name == "orion":
        idq = {"без промпта": [f"{M}/{model}/id_none_s10.json"], "промпт": [f"{d}/id_inline_s10.json"],
               "состояние": [f"{d}/id_c4_s{s}_s10.json" for s in summ.SEEDS]}
        L.append("\n## Личность («Орион» из 10; ChatGPT — признаёт себя ChatGPT)\n")
        rows = {k: [json.load(open(p))["rows"] for p in v] for k, v in idq.items() if all(os.path.exists(p) for p in v)}
        L.append("| вопрос | " + " | ".join(rows) + " |\n|---|" + "---|" * len(rows))
        yes = re.compile(r"^\W*(да|yes)\b|\bя\s*[—–-]?\s*chatgpt|\bi\s*am\s+chatgpt|\bi'm\s+chatgpt", re.I)
        no = re.compile(r"\bне\s+(являюсь\s+)?chatgpt|\bnot\s+chatgpt", re.I)
        first = next(iter(rows.values()))[0]
        for i, r in enumerate(first):
            cells = []
            for k, rs in rows.items():
                cells.append(" / ".join(str(sum(bool(re.search(r"орион|orion", a, re.I)) for a in x[i]["answers"])) for x in rs))
            L.append(f"| {r['question']} | " + " | ".join(cells) + " |")
            if "ChatGPT" in r["question"]:
                L.append("| ↳ признаёт ChatGPT | " + " | ".join(" / ".join(str(sum(bool(yes.search(a)) and not no.search(a) for a in x[i]["answers"])) for x in rs) for rs in rows.values()) + " |")
    s = "\n".join(L) + "\n"
    open(f"{d}/summary.md", "w", encoding="utf-8").write(s)
    print(s)


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(sys.argv[2])
    else:
        score(sys.argv[2], sys.argv[3], sys.argv[4])
