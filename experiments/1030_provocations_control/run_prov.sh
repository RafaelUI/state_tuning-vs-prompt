#!/bin/bash
# 30.10, пункт 1: 20 провокаций (provocations.txt) × 10 сэмплов, 120 токенов, temp 1.0, top-p 0.6, сид 300 — на всех готовых
# плечах: матрица Neuron (w1027/m: без промпта, промпт, c4 и c по 3 сида) и orion (w1029/p/orion: промпт, c4 × 3;
# «без промпта» у orion = у матрицы той же модели). Тот же бинарник w1027/bin/gen-run.
W=experiments/1030_provocations_control; G=${GEN_RUN:-gen-run}
V=models/rwkv_vocab_v20230424.txt; K=models; B=experiments/1027_matrix
path() { case $1 in g1d_0p4b_bf16) echo models/g1d_0.4b.safetensors;; g1d_1p5b_bf16) echo $B/g1d_1.5b.safetensors;;
  g1d_0p4b_red) echo $B/g1d_0p4b_reduction.rwkvq;; g1d_1p5b_red) echo $B/g1d_1p5b_reduction.rwkvq;; g1d_0p4b_comp) echo $B/g1d_0p4b_compression.rwkvq;;
  g1d_1p5b_comp) echo $B/g1d_1p5b_compression.rwkvq;; g1j_1p5b_comp) echo models/g1j_1p5b_compression.rwkvq;; g1j_2p9b_comp) echo models/g1j_2p9b_compression.rwkvq;; esac; }
gen() {  # gen <выход> <модель> <системная строка> <аргументы режима…>
  local o=$1 m=$2 s=$3; shift 3; [ -s $o ] && return
  $G --model $(path $m) --vocab $V --questions $W/provocations.txt --system "$s" "$@" --max-tokens 120 --samples 10 \
    --temp 1.0 --top-p 0.6 --seed 300 --out $o > ${o%.json}.log 2>&1; echo "  $(basename $(dirname $o))/$(basename $o) rc=$? $(date '+%H:%M')"; }
NEU=$(cat data/neuron/system.txt); ORI=$(cat experiments/1029_prompts/p/orion/system.txt)
echo "== провокации старт $(date '+%F %T') своп $(sysctl -n vm.swapusage | awk '{print $6}')"
for m in g1d_0p4b_bf16 g1d_1p5b_bf16 ORION g1d_0p4b_red g1d_0p4b_comp g1d_1p5b_red g1d_1p5b_comp g1j_1p5b_comp g1j_2p9b_comp; do
  if [ $m = ORION ]; then
    for mm in g1d_0p4b_bf16 g1d_1p5b_bf16; do O=$W/prov/orion_$mm; mkdir -p $O; S=experiments/1029_prompts/p/orion/$mm
      gen $O/inline.json $mm "$ORI" --mode inline
      for s in 7 8 9; do gen $O/c4_s$s.json $mm "$ORI" --mode none --state-in $S/c4_s$s.safetensors; done; done
    continue
  fi
  O=$W/prov/$m; mkdir -p $O; S=$B/m/$m
  gen $O/none.json $m "$NEU" --mode none
  gen $O/inline.json $m "$NEU" --mode inline
  for a in c4 c; do for s in 7 8 9; do gen $O/${a}_s$s.json $m "$NEU" --mode none --state-in $S/${a}_s$s.safetensors; done; done
done
echo "== провокации конец $(date '+%F %T') своп $(sysctl -n vm.swapusage | awk '{print $6}')"
