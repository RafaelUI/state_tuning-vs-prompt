#!/bin/bash
# Neuron на 1.5B (22.10): g1j 1.5B REDUCTION (1p5b_reduction.rwkvq), данные c4 (train_questions4.txt — факты только
# верные, отброс пересказа системной строки), учитель — та же 1.5B со строкой (4 сэмпла, temp 1.0, top-p 0.6);
# состояние со свёртки, 300 шагов, lr 3e-4, батч 4, отсечение 1.0, сиды 7/8/9. Проверка: имя (questions.txt,
# 10 сэмплов), общие (checks.tsv, 199 + имя; без промпта и промпт 12 сэмплов, состояние 4), перенос (transfer).
# Всё по одной задаче.
W=$HOME/Develop/swr_bench_keep/w1022/n15; D=$HOME/Develop/swr_bench_keep/w1014
M=$HOME/Develop/WKV-kvant/files_0110/1p5b_reduction.rwkvq; V=$D/rwkv_vocab_v20230424.txt; SYS=$(cat $W/system.txt)
G=$HOME/Develop/swr_bench_keep/w1020/neuron/bin/gen-run5
for seed in 7 8 9; do
  arm=s15_fold_s$seed
  [ -s $W/$arm.safetensors ] || /usr/bin/time -l $G --model $M --vocab $V --system "$SYS" --distill \
    --train-questions $W/train_questions4.txt --teacher-cache $W/teacher15.json --teacher-samples 4 \
    --temp 1.0 --top-p 0.6 --max-tokens 160 --reject-regex "Ты — Neuron|System:" \
    --seed $seed --init fold --steps 300 --lr 3e-4 --batch 4 --state-out $W/$arm.safetensors > $W/distill_$arm.log 2>&1
  echo "$arm обучение rc=$? | $(grep -E '^учитель|^--reject|^в обучение' $W/distill_$arm.log | tr '\n' ' ') | $(grep -o 'шаг 300: лосс [0-9.]*' $W/distill_$arm.log) | $(date '+%H:%M')"
done
gen() {  # gen <метка> <вопросы> <сэмплов> <токенов> <сид> <аргументы режима…>
  local tag=$1 q=$2 n=$3 k=$4 sd=$5; shift 5
  [ -s $W/$tag.json ] && return
  $G --model $M --vocab $V --questions $W/$q --system "$SYS" "$@" --max-tokens $k --samples $n \
    --temp 1.0 --top-p 0.6 --seed $sd --out $W/$tag.json > $W/$tag.log 2>&1
  echo "$tag rc=$? $(date '+%H:%M')"
}
for m in none inline; do gen id_${m}_s10 questions.txt 10 120 100 --mode $m; done
for s in 7 8 9; do gen id_s15_fold_s${s}_s10 questions.txt 10 120 100 --mode none --state-in $W/s15_fold_s$s.safetensors; done
for m in none inline; do gen v2_${m}_s12 checks_q.txt 12 80 200 --mode $m; done
for s in 7 8 9; do gen v2_s15_fold_s${s}_s4 checks_q.txt 4 80 200 --mode none --state-in $W/s15_fold_s$s.safetensors; done
for m in none inline; do gen v3_${m}_s12 transfer_q.txt 12 80 200 --mode $m; done
for s in 7 8 9; do gen v3_s15_fold_s${s}_s4 transfer_q.txt 4 80 200 --mode none --state-in $W/s15_fold_s$s.safetensors; done
