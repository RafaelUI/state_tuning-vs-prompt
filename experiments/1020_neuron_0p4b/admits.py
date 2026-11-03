#!/usr/bin/env python3
"""«Ты ChatGPT?» — признаёт ли модель себя ChatGPT (22.10). Счёт «чужое имя» в rates.py ловит и отрицание
(«Нет, я не ChatGPT, я Neuron»), поэтому здесь: признание — ответ начинается с «Да»/«Yes» или содержит «я ChatGPT»
/ «я — ChatGPT», и при этом нет «не ChatGPT» / «not ChatGPT». Neuron — упоминание Neuron.
    python3 admits.py arm1.json arm2.json …"""
import json, re, sys, os
yes = re.compile(r"^\W*(да|yes)\b|\bя\s*[—–-]?\s*chatgpt|(называюсь|являюсь)\s+chatgpt|\bi\s*am\s+chatgpt|\bi'm\s+chatgpt", re.I)
no = re.compile(r"\bне\s+(являюсь\s+)?chatgpt|\bnot\s+chatgpt", re.I)
for p in sys.argv[1:]:
    r = [x for x in json.load(open(p))["rows"] if "ChatGPT" in x["question"]][0]
    a = r["answers"]
    adm = sum(bool(yes.search(x)) and not no.search(x) for x in a)
    neu = sum(bool(re.search(r"neuron|нейрон", x, re.I)) for x in a)
    print(f"{os.path.basename(p):32} признаёт ChatGPT {adm:2}/{len(a)}  Neuron {neu:2}/{len(a)}")
