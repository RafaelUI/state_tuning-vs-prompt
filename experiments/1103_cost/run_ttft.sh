#!/bin/bash
# 03.11: таблица стоимости — время до первого токена: промпт в строке (inline) против свёрнутого промпта (fold = обученное
# состояние по цене подключения). 50 вопросов checks_q × 2 сэмпла = 100 запросов, --max-tokens 1 (только префилл + 1 токен).
# Строки: короткая (Neuron, 26 токенов) и длинная (правила поддержки, 1206 токенов, system_long.txt). Модели: g1d 0.4B / 1.5B bf16,
# g1j 2.9B COMPRESSION. Плечи вперемежку, 3 круга, порядок сдвигается. Бинарник w1101/bin_cls/gen-run (c332453). Машина свободна.
W=experiments/1103_cost; G=${GEN_RUN:-gen-run}
V=models/rwkv_vocab_v20230424.txt
SHORT=$(cat data/neuron/system.txt); LONG=$(cat $W/system_long.txt)
path() { case $1 in 0p4b) echo models/g1d_0.4b.safetensors;; 1p5b) echo models/g1d_1.5b.safetensors;;
  2p9b) echo models/g1j_2p9b_compression.rwkvq;; esac; }
sw() { sysctl -n vm.swapusage | awk '{print $6}'; }
echo "== ttft старт $(date '+%F %T') своп $(sw); ${GEN_RUN:-gen-run}"
for m in 0p4b 1p5b 2p9b; do for sy in short long; do
  S=$([ $sy = short ] && echo "$SHORT" || echo "$LONG")
  for r in 1 2 3; do
    arms="inline fold"; [ $((r % 2)) = 0 ] && arms="fold inline"
    for a in $arms; do
      o=$W/${m}_${sy}_${a}_r$r; s0=$(sw)
      /usr/bin/time -l $G --model $(path $m) --vocab $V --questions $W/q50.txt --system "$S" --mode $a --max-tokens 1 --samples 2 \
        --temp 1.0 --top-p 0.6 --seed 200 --out $o.json > $o.log 2>&1
      echo "$m $sy $a r$r rc=$? | $(grep -o 'загрузка [0-9.]* с' $o.log) | $(grep -o 'всего [0-9.]* с' $o.log) | пик $(grep 'peak memory footprint' $o.log | awk '{printf "%.2f ГБ", $1/1e9}') | своп $s0 -> $(sw)"
    done
  done
done; done
echo "== ttft конец $(date '+%F %T') своп $(sw)"
