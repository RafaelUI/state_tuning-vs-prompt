#!/usr/bin/env python3
"""01.11: классификация новостей Lenta.ru (data-silence/lenta.ru_2) для g1d 0.4B — инструкция «рубрика одним словом».
Решения Алексея 01.11: оба плеча (дистилляция с фильтром по золотой рубрике и обучение на золотых метках), одна статья ×3 длины,
только 0.4B. Рубрики: Спорт, Экономика, Наука и техника, Культура, Путешествия -> спорт, экономика, наука, культура, путешествия.
Проверка: 30 статей на рубрику (≥3000 символов), каждая обрезана по предложению до ~100 / ~400 / ~1000 токенов (одни и те же
статьи — парное сравнение по длине). Обучение: 50 других статей на рубрику, длина — по кругу одна из трёх отсечек.
Выход: system.txt, train.txt (вид fact: регэксп рубрики :: текст), test_{short,mid,long}_q.txt + test.tsv, gold_answers.tsv."""
import csv, gzip, json, random, re, sys
sys.path.insert(0, "../../lib"); from world_tokenizer import RWKV_WORLD_TOKENIZER
tok = RWKV_WORLD_TOKENIZER("../../models/rwkv_vocab_v20230424.txt")
csv.field_size_limit(10**9)
TOP = {"Спорт": "спорт", "Экономика": "экономика", "Наука и техника": "наука", "Культура": "культура", "Путешествия": "путешествия"}
TARGET = {"short": 100, "mid": 400, "long": 1000}
SYSTEM = ("Ты — классификатор новостей. Пользователь пришлёт текст новости. Ответь одним словом — рубрикой из списка: "
          "спорт, экономика, наука, культура, путешествия. Больше ничего не пиши.")
def clean(t): return re.sub(r"\s+", " ", t.replace(" ", " ")).strip()
def cut(text, ntok):
    """Самый длинный префикс по границе предложения, не длиннее ntok токенов (не меньше одного предложения)."""
    sents = re.split(r"(?<=[.!?…»])\s*(?=[А-ЯЁA-Z«\"])", text)
    out = sents[0]
    for s in sents[1:]:
        cand = out + " " + s
        if len(tok.encode(cand)) > ntok: break
        out = cand
    return out
by = {t: [] for t in TOP}
with gzip.open("final_lenta_new.csv.gz", "rt", encoding="utf-8", newline="") as f:
    for r in csv.DictReader(f):
        if r["topic"] in TOP and " :: " not in r["news"]:
            by[r["topic"]].append(clean(r["news"]))
rng = random.Random(1101)
test, train = [], []
for t, lab in TOP.items():
    docs = by[t]; rng.shuffle(docs)
    longd = [d for d in docs if len(d) >= 3000][:30]
    rest = [d for d in docs if d not in set(longd) and len(d) >= 400][:50]
    assert len(longd) == 30 and len(rest) == 50, t
    for d in longd: test.append((lab, {k: cut(d, n) for k, n in TARGET.items()}))
    for i, d in enumerate(rest): train.append((lab, cut(d, list(TARGET.values())[i % 3])))
rng.shuffle(test); rng.shuffle(train)
open("system.txt", "w", encoding="utf-8").write(SYSTEM)
rx = lambda lab: rf"^[\s*«\"]*{lab}[\s*»\".!]*$"   # ровно одно слово-рубрика (регистр не важен), допускаются кавычки/точка
with open("train.txt", "w", encoding="utf-8") as f:
    f.write("# 01.11: классификация Lenta.ru, 5 рубрик × 50 статей (длина по кругу ~100/~400/~1000 токенов); вид fact: ответ — ровно рубрика\n")
    for lab, d in train: f.write(f"fact: {rx(lab)} :: {d}\n")
with open("gold_answers.tsv", "w", encoding="utf-8") as f:   # для плеча «золотые метки»: вопрос \t ответ
    for lab, d in train: f.write(f"{d}\t{lab}\n")
with open("test.tsv", "w", encoding="utf-8") as f:
    f.write("# метка\tдлина\tтекст\n")
    for k in TARGET:
        with open(f"test_{k}_q.txt", "w", encoding="utf-8") as q:
            for lab, v in test: q.write(v[k] + "\n"); f.write(f"{lab}\t{k}\t{v[k]}\n")
for k in TARGET:
    L = sorted(len(tok.encode(v[k])) for _, v in test); print(k, "токенов: мин", L[0], "медиана", L[len(L)//2], "макс", L[-1])
L = sorted(len(tok.encode(d)) for _, d in train); print("обучение: токенов мин", L[0], "медиана", L[len(L)//2], "макс", L[-1], "; статей", len(train))
print("регэксп пример:", rx("спорт"))
