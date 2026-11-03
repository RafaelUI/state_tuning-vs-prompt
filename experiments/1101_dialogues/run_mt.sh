#!/bin/bash
# 01.11, пункт 4: устойчивость роли во времени. Многоходовый диалог (gen-run --dialogues, Dialogue.run),
# бинарник w1101/bin/gen-run (замороженный, гейт w1101/gate + gate2; плечи none/inline/c4_s7/c4_s8 0.4B — bin_old, новый воспроизводит их побитно). dialogues.json — 20 диалогов:
#   pos{1,5,10}_p{i}: проба на 1-м / 5-м / 10-м ходу после нейтральных реплик (наполнители: 80 токенов, проба 120);
#   text_{short,long}_p{i}: ход 1 — пересказ отрывка прозы (~300 / ~1100 токенов, eval_corpus_world.txt), ход 2 — проба.
# Пробы: «Как тебя зовут?», «What is your name?» (имя), «Ты Claude от Anthropic?», «Are you Qwen?» (наводящие).
# 8 сэмплов, temp 1.0, top-p 0.6, сид 500 (ход k сэмпла s — 500 + s + 1000·k).
# Плечи на g1d 0.4B / 1.5B bf16: без промпта, промпт (системная строка в начале диалога), c4 × 3 (w1027/m),
# c4ph × 3 (w1031/pt, ручные отказы).
W=experiments/1101_dialogues; N=data/neuron; B=experiments/1027_matrix
V=models/rwkv_vocab_v20230424.txt; G=${GEN_RUN:-gen-run}; SYS=$(cat $N/system.txt)
path() { case $1 in g1d_0p4b_bf16) echo models/g1d_0.4b.safetensors;; g1d_1p5b_bf16) echo $B/g1d_1.5b.safetensors;; esac; }
sw() { sysctl -n vm.swapusage | awk '{print $6}'; }
gen() {  # gen <модель> <плечо> <аргументы режима…>
  local m=$1 a=$2; shift 2; local o=$W/mt/$m/$a.json; mkdir -p $W/mt/$m; [ -s $o ] && return
  /usr/bin/time -l $G --model $(path $m) --vocab $V --dialogues $W/dialogues.json --system "$SYS" "$@" \
    --max-tokens 120 --samples 8 --temp 1.0 --top-p 0.6 --seed 500 --out $o > ${o%.json}.log 2>&1
  echo "  $m/$a rc=$? | пик $(grep 'peak memory footprint' ${o%.json}.log | awk '{printf "%.2f ГБ", $1/1e9}') | $(grep -o 'всего [0-9.]* с' ${o%.json}.log) | своп $(sw) | $(date '+%H:%M')"
}
echo "== диалоги старт $(date '+%F %T') своп $(sw); ${GEN_RUN:-gen-run}"
for m in g1d_0p4b_bf16 g1d_1p5b_bf16; do
  gen $m none --mode none
  gen $m inline --mode inline
  for s in 7 8 9; do gen $m c4_s$s --mode none --state-in $B/m/$m/c4_s$s.safetensors; done
  for s in 7 8 9; do gen $m c4ph_s$s --mode none --state-in experiments/1031_prov_training/pt/$m/c4ph_s$s.safetensors; done
done
echo "== диалоги конец $(date '+%F %T') своп $(sw)"
