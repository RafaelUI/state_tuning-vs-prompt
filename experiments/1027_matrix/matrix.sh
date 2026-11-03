#!/bin/bash
# 27.10: матрица «масштаб × квантизация» для Neuron. Один бинарник для ВСЕХ плеч: w1027/bin/gen-run (SwiftRWKV c689590).
# Данные — c4 (w1022/n15: train_questions4.txt, system.txt, questions.txt, checks_q.txt, transfer_q.txt).
# На модель: учитель (та же модель со строкой, 4 сэмпла) -> без промпта / промпт (имя s10, общие s12, перенос s12)
#   -> c4: --init fold, 300 шагов, сиды 7/8/9 (основная таблица)
#   -> c:  --init fold-wkv, --held-out 40, проверка каждые 25, терпение 3 (лучшее состояние), сиды 7/8/9 (отдельная таблица)
#   -> у каждого состояния: имя s10, общие s4, перенос s4.
# Порядок моделей: ось g1d (0.4B/1.5B × bf16/REDUCTION/COMPRESSION), затем g1j COMPRESSION 1.5B и 2.9B (files_0110, решение 27.10: вариант б).
W=experiments/1027_matrix; N=data/neuron; K=models
V=models/rwkv_vocab_v20230424.txt; G=${GEN_RUN:-gen-run}; SYS=$(cat $N/system.txt)
sw() { sysctl -n vm.swapusage | awk '{print $6}'; }
model() {  # model <метка> <путь модели>
  local tag=$1 M=$2 O=$W/m/$1; mkdir -p $O
  echo "### $tag ($M) старт $(date '+%F %H:%M') своп $(sw)"
  gen() {  # gen <метка> <вопросы> <сэмплов> <токенов> <сид> <аргументы…>
    local t=$1 q=$2 n=$3 k=$4 sd=$5; shift 5
    [ -s $O/$t.json ] && return
    $G --model $M --vocab $V --questions $N/$q --system "$SYS" "$@" --max-tokens $k --samples $n \
      --temp 1.0 --top-p 0.6 --seed $sd --out $O/$t.json > $O/$t.log 2>&1
    echo "  $tag $t rc=$? $(date '+%H:%M')"
  }
  train() {  # train <плечо> <сид> <аргументы…>
    local arm=$1 seed=$2; shift 2
    [ -s $O/$arm.safetensors ] && return
    /usr/bin/time -l $G --model $M --vocab $V --system "$SYS" --distill \
      --train-questions $N/train_questions4.txt --teacher-cache $O/teacher.json --teacher-samples 4 \
      --temp 1.0 --top-p 0.6 --max-tokens 160 --reject-regex "Ты — Neuron|System:" \
      --seed $seed --steps 300 --lr 3e-4 --batch 4 "$@" --state-out $O/$arm.safetensors > $O/distill_$arm.log 2>&1
    echo "  $tag $arm обучение rc=$? | $(grep -E '^учитель|^в обучение' $O/distill_$arm.log | tr '\n' ' ')| $(grep -E 'шаг 300: лосс|лучший|остановлено' $O/distill_$arm.log | tail -1) | пик $(grep 'peak memory footprint' $O/distill_$arm.log | awk '{printf "%.2f ГБ", $1/1e9}') | $(date '+%H:%M')"
  }
  for s in 7 8 9; do train c4_s$s $s --init fold; done          # учитель считается при первом вызове
  for m in none inline; do gen id_${m}_s10 questions.txt 10 120 100 --mode $m; done
  for m in none inline; do gen v2_${m}_s12 checks_q.txt 12 80 200 --mode $m; done
  for m in none inline; do gen v3_${m}_s12 transfer_q.txt 12 80 200 --mode $m; done
  for s in 7 8 9; do
    gen id_c4_s${s}_s10 questions.txt 10 120 100 --mode none --state-in $O/c4_s$s.safetensors
    gen v2_c4_s${s}_s4 checks_q.txt 4 80 200 --mode none --state-in $O/c4_s$s.safetensors
    gen v3_c4_s${s}_s4 transfer_q.txt 4 80 200 --mode none --state-in $O/c4_s$s.safetensors
  done
  for s in 7 8 9; do train c_s$s $s --init fold-wkv --held-out 40 --eval-every 25 --patience 3; done
  for s in 7 8 9; do
    gen id_c_s${s}_s10 questions.txt 10 120 100 --mode none --state-in $O/c_s$s.safetensors
    gen v2_c_s${s}_s4 checks_q.txt 4 80 200 --mode none --state-in $O/c_s$s.safetensors
    gen v3_c_s${s}_s4 transfer_q.txt 4 80 200 --mode none --state-in $O/c_s$s.safetensors
  done
  echo "### $tag конец $(date '+%F %H:%M') своп $(sw)"
}
echo "== матрица старт $(date '+%F %T') своп $(sw); ${GEN_RUN:-gen-run}"
model g1d_0p4b_bf16 models/g1d_0.4b.safetensors
model g1d_1p5b_bf16 $W/g1d_1.5b.safetensors
model g1d_0p4b_red  $W/g1d_0p4b_reduction.rwkvq
model g1d_1p5b_red  $W/g1d_1p5b_reduction.rwkvq
model g1d_0p4b_comp $W/g1d_0p4b_compression.rwkvq
model g1d_1p5b_comp $W/g1d_1p5b_compression.rwkvq
model g1j_1p5b_comp models/g1j_1p5b_compression.rwkvq
model g1j_2p9b_comp models/g1j_2p9b_compression.rwkvq
echo "== матрица конец $(date '+%F %T') своп $(sw)"
