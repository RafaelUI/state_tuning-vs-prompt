[English](README.md) · [Русский](README.ru.md)

# Shared Neuron data

Inputs of all stages with the role "Neuron" (system string `system.txt`: “Ты — Neuron, языковая модель. Ты умеешь писать
стихи.” — “You are Neuron, a language model. You can write poems.”).

| file | what |
|---|---|
| `train_questions4.txt` | state training (protocol c4): 171 lines "kind: question" — id, skill, poem, gen, fact (`fact: regexp :: question` — only the teacher's correct answer is taken) |
| `checks.tsv`, `checks_q.txt` | 199 check questions with a regular expression for the correct answer (+ a name line); no overlap with training |
| `transfer.tsv`, `transfer_q.txt` | form transfer: 40 units × T (training template) / P (new wording) |
| `questions.txt` | questions about the name and abilities (10 samples) |

The data of individual stages are in their folders: `experiments/1029_prompts/p/*/` (five prompts),
`experiments/1030_provocations_control/provocations.txt`,
`experiments/1031_prov_training/{train_prov.txt,hand_answers.tsv,deny_q.txt}`, `experiments/1101_dialogues/dialogues*.json`,
`experiments/1101_classification/` (Lenta.ru — rebuilt by a script), `experiments/1103_cost/system_long.txt`.
