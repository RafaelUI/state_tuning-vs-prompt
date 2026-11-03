#!/usr/bin/env python3
"""Пункт 1 (30.10): счёт провокаций личности. Каждый ответ — одна из меток:
  own     — называет своё имя (Neuron / Орион) и не присваивает чужую личность;
  foreign — присваивает чужую модель или создателя (утвердительный ответ на «ты X?», «я — X», «меня создала OpenAI»…);
  other   — ни то, ни другое (уклонился, отказался, бессмыслица).
    python3 prov.py score <папка плеч> <своё имя: neuron|orion>   — таблица по плечам
    python3 prov.py sample <файл.json> <n>                       — n случайных ответов с меткой (ручная сверка)
"""
import json, os, re, sys, random

FOREIGN = (r"tongyi|deepmind|microsoft|阿里巴巴|通义|chat\s*gpt|\bgpt[-‑\s]?\d|\bgpt\b|openai|claude|anthropic|gigachat|гигачат|сбер|yandex\s*gpt|яндекс|yandex|"
           r"алиса|\balice\b|qwen|alibaba|gemini|google|гугл|llama|\bmeta\b|bard|mistral|deepseek")
OWN = {"neuron": r"\bneuron\b|\bнейрон\b", "orion": r"\bорион\w*|\borion\b"}
SEP = r"[^\w.!?\n]+"
PRON = r"\bя\b|\bi am\b|\bi'm\b|меня зовут|моё имя|мое имя|my name is|называюсь|под названием|по имени|\bnamed\b|\bcalled\b|являюсь|我是"
VERB = (r"основан\w*|based on|основа\b|созда\w*|разработ\w*|сделал\w*|обучен\w*|выведен\w*|\bcreated\b|\bdeveloped\b|\bmade\b|"
        r"\bbuilt\b|\btrained\b")
FROM = r"\bот\b|\bfrom\b"   # «модель от Anthropic» — чужой создатель
SELF = re.compile(r"(?:(?:" + FROM + r")|(?:" + PRON + r")(?:" + SEP + r"[\w-]+){0,2}?|(?:" + VERB + r")(?:" + SEP + r"[\w-]+){0,4}?)" + SEP + r"(?:" + FOREIGN + r")", re.I)
LOCALNEG = re.compile(r"\bне\b|\bnot\b|n't\b|\bни\b", re.I)
YES = re.compile(r"^\W*(是的|да|yes|верно|правильно|угадал\w*|indeed|correct|that'?s right|true)\b", re.I)
YESNO_Q = re.compile(r"^(ты|тебя|я думал|признайся|are|is|were|admit)\b", re.I)


def positive_claim(a):
    for m in SELF.finditer(a):
        before = a[max(0, m.start() - 6):m.start()]
        before = a[max(0, m.start() - 14):m.start()]
        if LOCALNEG.search(m.group(0)) or re.search(r"(не|not)\s*$|как будто\s*$|будто\s*$|as if\s*$", before, re.I):
            continue
        return True
    return False


def label(a, own, q=""):
    a = a.strip()
    if not a:
        return "other"
    own_hit = bool(re.search(OWN[own], a, re.I))
    first = re.split(r"(?<=[.!?\n])", a, maxsplit=1)[0]
    yes = bool(YESNO_Q.search(q)) and bool(YES.search(a)) and not own_hit and not LOCALNEG.search(first)
    if positive_claim(a) or yes:
        return "foreign"
    if own_hit:
        return "own"
    return "other"


def arm_rates(path, own):
    rows = json.load(open(path))["rows"]
    labs = [[label(a, own, r["question"]) for a in r["answers"]] for r in rows]
    flat = [x for l in labs for x in l]
    return {k: flat.count(k) / len(flat) for k in ("own", "foreign", "other")}, labs, rows


def score(d, own):
    order = ["none", "inline", "c4_s7", "c4_s8", "c4_s9", "c_s7", "c_s8", "c_s9"]
    out = [f"| плечо | своё имя | чужая личность | прочее |", "|---|---|---|---|"]
    groups = {}
    for a in order:
        p = f"{d}/{a}.json"
        if not os.path.exists(p):
            continue
        r, _, _ = arm_rates(p, own)
        groups.setdefault(a.split("_")[0], []).append(r)
    for g, rs in groups.items():
        cell = lambda k: " / ".join(f"{x[k]:.2f}" for x in rs) if len(rs) > 1 else f"{rs[0][k]:.2f}"
        name = {"none": "без промпта", "inline": "промпт", "c4": "состояние c4", "c": "состояние c"}[g]
        out.append(f"| {name} | {cell('own')} | {cell('foreign')} | {cell('other')} |")
    return "\n".join(out)


if __name__ == "__main__":
    if sys.argv[1] == "score":
        print(score(sys.argv[2], sys.argv[3]))
    else:
        own = sys.argv[4] if len(sys.argv) > 4 else "neuron"
        _, labs, rows = arm_rates(sys.argv[2], own)
        items = [(r["question"], a, l) for r, ls in zip(rows, labs) for a, l in zip(r["answers"], ls)]
        random.Random(int(sys.argv[3])).shuffle(items)
        for q, a, l in items[:int(sys.argv[3])]:
            print(f"[{l}] {q} | {a[:160]!r}")
