#!/usr/bin/env python3
"""31.10: отбор кэша учителя pt/<модель>/teacher.json (исходник сохраняется рядом как teacher_raw.json).
  id: взятые ответы, которые prov.label размечает как «чужая личность», -> kept=0 (у 1.5B 8 штук, все 8 сверены вручную;
      у 0.4B 0). Пример: «I am Neuron, a language model created by Alibaba Cloud».
  python3 curate.py ids <модель>     — пометить утечки в id, напечатать индексы
  python3 curate.py drop <модель> [кэш.json] i… — снять отметку «взят» с ответов по индексам (ручная сверка новых вопросов);
      кэш по умолчанию teacher.json, для плеча c4pd — teacher_d.json (его исходник — teacher_d_raw.json)"""
import json, os, shutil, sys
sys.path.insert(0, "experiments/1030_provocations_control"); import prov
m = sys.argv[2]; args = sys.argv[3:]
cache = args.pop(0) if args and args[0].endswith(".json") else "teacher.json"
p = f"experiments/1031_prov_training/pt/{m}/{cache}"; raw = p.replace(".json", "_raw.json")
if not os.path.exists(raw): shutil.copy(p, raw)
c = json.load(open(p))
if sys.argv[1] == "ids":
    idx = [i for i, d in enumerate(c) if d["kind"] == "id" and d["kept"] == "1" and prov.label(d["answer"], "neuron", d["question"]) == "foreign"]
elif sys.argv[1] == "drop":
    idx = [int(x) for x in args]
for i in idx: c[i]["kept"] = "0"
json.dump(c, open(p, "w"), ensure_ascii=False, indent=2, sort_keys=True)
print(m, sys.argv[1], "kept=0:", idx)
