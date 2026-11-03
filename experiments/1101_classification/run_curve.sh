#!/bin/bash
# 02.11: кривая по объёму данных (решение Алексея 02.11): состояние против LoRA r16 на классификации g1d 0.4B bf16,
# 25 / 50 / 125 статей (по 5 / 10 / 25 на рубрику, вложенные подмножества train.txt — curve/train_n*.txt); точка 250 — основной
# прогон (cg, lora lr 3e-4). Протокол тот же, что у точки 250: 300 шагов, батч 4, отсечение 1.0; состояние — fold, lr 3e-4;
# LoRA — lr 3e-4 (выбран по валидации на 250, на малых N не перебирается — оговорка). Сиды 7/8/9.
# Проверка — 150 статей ~400 токенов (test_mid), 4 сэмпла, сид 200, 16 токенов (другие длины — в основном прогоне).
# Бинарник w1101/bin_cls/gen-run (c332453).
W=experiments/1101_dialogues; C=experiments/1101_classification; G=${GEN_RUN:-gen-run}; M=models/g1d_0.4b.safetensors
V=models/rwkv_vocab_v20230424.txt; SYS=$(cat $C/system.txt); O=$C/curve/g1d_0p4b_bf16; mkdir -p $O
sw() { sysctl -n vm.swapusage | awk '{print $6}'; }
train() {  # train <n> <плечо> <сид> <аргументы…>
  local n=$1 a=$2 s=$3; shift 3; local f=$O/n${n}_${a}_s$s.safetensors; [ -s $f ] && return
  /usr/bin/time -l $G --model $M --vocab $V --system "$SYS" --distill --train-questions $C/curve/train_n$n.txt --teacher-cache $C/teacher_g.json \
    --teacher-samples 4 --temp 1.0 --top-p 0.6 --max-tokens 16 --seed $s --steps 300 --lr 3e-4 --batch 4 "$@" > ${f%.safetensors}.log 2>&1
  local rc=$?; [ -s $f ] || { echo "ОШИБКА обучения n$n $a s$s (rc $rc): $(tail -2 ${f%.safetensors}.log | tr '\n' ' ')"; exit 3; }
  echo "  n$n $a s$s rc=$rc | $(grep -E '^в обучение' ${f%.safetensors}.log | cut -c1-60) | $(grep -oE 'шаг 300: лосс [0-9.]*' ${f%.safetensors}.log | tail -1) | пик $(grep 'peak memory footprint' ${f%.safetensors}.log | awk '{printf "%.2f ГБ", $1/1e9}') | своп $(sw) | $(date '+%H:%M')"
}
gen() {  # gen <выход> <аргументы…>
  local o=$1; shift; [ -s $o ] && return
  $G --model $M --vocab $V --questions $C/test_mid_q.txt --system "$SYS" --mode none "$@" --max-tokens 16 --samples 4 --temp 1.0 --top-p 0.6 --seed 200 \
    --out $o > ${o%.json}.log 2>&1; echo "  $(basename $o) rc=$? $(date '+%H:%M')"
}
echo "== кривая по данным старт $(date '+%F %T') своп $(sw); ${GEN_RUN:-gen-run}"
for n in 25 50 125; do
  for s in 7 8 9; do
    train $n cg $s --init fold --state-out $O/n${n}_cg_s$s.safetensors
    gen $O/mid_n${n}_cg_s$s.json --state-in $O/n${n}_cg_s$s.safetensors
    train $n lora $s --train lora --weights-out $O/n${n}_lora_s$s.safetensors
    gen $O/mid_n${n}_lora_s$s.json --lora-in $O/n${n}_lora_s$s.safetensors
  done
done
echo "== кривая по данным конец $(date '+%F %T') своп $(sw)"
