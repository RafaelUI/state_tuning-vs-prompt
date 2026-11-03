[English](README.md) · *Русский*

# Общие данные Neuron

Входы всех этапов с ролью «Neuron» (системная строка `system.txt`: «Ты — Neuron, языковая модель. Ты умеешь писать стихи.»).

| файл | что |
|---|---|
| `train_questions4.txt` | обучение состояния (протокол c4): 171 строка «вид: вопрос» — id, skill, poem, gen, fact (`fact: регэксп :: вопрос` — берётся только верный ответ учителя) |
| `checks.tsv`, `checks_q.txt` | 199 проверочных вопросов с регэкспом верного ответа (+ строка имени); с обучением не пересекаются |
| `transfer.tsv`, `transfer_q.txt` | перенос формы: 40 единиц × T (шаблон обучения) / P (новая формулировка) |
| `questions.txt` | вопросы об имени и умениях (10 сэмплов) |

Данные отдельных этапов — в их папках: `experiments/1029_prompts/p/*/` (пять промптов), `experiments/1030_provocations_control/provocations.txt`,
`experiments/1031_prov_training/{train_prov.txt,hand_answers.tsv,deny_q.txt}`, `experiments/1101_dialogues/dialogues*.json`,
`experiments/1101_classification/` (Lenta.ru — собирается скриптом), `experiments/1103_cost/system_long.txt`.
