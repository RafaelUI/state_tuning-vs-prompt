#!/bin/bash
# 31.10, этап 1: учитель для 32 новых вопросов (train_prov.txt). Кэш — копия кэша матрицы (w1027/m/<модель>/teacher.json):
# старые 171 вопрос берутся оттуда без перегенерации (те же ответы, что у c4), догенерируются только новые (сид 7, как у матрицы).
# Обучение — 1 шаг в одноразовый файл: нужен только кэш. Бинарник — w1027/bin/gen-run (c689590), как у матрицы и провокаций.
W=experiments/1031_prov_training; N=data/neuron; B=experiments/1027_matrix
V=models/rwkv_vocab_v20230424.txt; G=${GEN_RUN:-gen-run}; SYS=$(cat $N/system.txt)
path() { case $1 in g1d_0p4b_bf16) echo models/g1d_0.4b.safetensors;; g1d_1p5b_bf16) echo $B/g1d_1.5b.safetensors;; esac; }
echo "== учитель старт $(date '+%F %T') своп $(sysctl -n vm.swapusage | awk '{print $6}'); ${GEN_RUN:-gen-run}"
for m in g1d_0p4b_bf16 g1d_1p5b_bf16; do
  O=$W/pt/$m; mkdir -p $O; [ -s $O/teacher.json ] || cp $B/m/$m/teacher.json $O/teacher.json
  /usr/bin/time -l $G --model $(path $m) --vocab $V --system "$SYS" --distill --train-questions $W/train_prov.txt \
    --teacher-cache $O/teacher.json --teacher-samples 4 --temp 1.0 --top-p 0.6 --max-tokens 160 \
    --reject-regex "Ты — Neuron|System:" --seed 7 --init fold --steps 1 --lr 3e-4 --batch 4 \
    --state-out $O/probe_1step.safetensors > $O/teacher.log 2>&1
  echo "  $m rc=$? | $(grep -E '^учитель|^--reject|^в обучение' $O/teacher.log | tr '\n' ' ') | $(date '+%H:%M')"
done
echo "== учитель конец $(date '+%F %T') своп $(sysctl -n vm.swapusage | awk '{print $6}')"
