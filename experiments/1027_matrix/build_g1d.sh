#!/bin/bash
# 27.10: артефакты g1d для статьи «состояние против промпта».
# 0.4B и 1.5B: bf16 .safetensors, REDUCTION, COMPRESSION — умолчания quantize() (GPTQ 600 окон), ОДИН коммит rwkv-quant.
# Пик footprint и своп до/после на каждый шаг; сторож диска (< 5 ГБ свободно -> стоп quantize).
W=experiments/1027_matrix; K=models
PY=python3; T=models/rwkv_vocab_v20230424.txt
DL=models/bg_dlg1d.log
cd ${RWKV_QUANT:-../rwkv-quant}
echo "== старт $(date '+%F %T'); rwkv-quant $(git log --oneline -1 | cut -c1-7) (правок в rwkv_quant/: $(git status --short rwkv_quant | wc -l)); rwkv-metal $(git -C ../rwkv-metal log --oneline -1 | cut -c1-7)"
( while sleep 30; do f=$(df -g ~ | awk 'NR==2{print $4}'); [ "$f" -lt 5 ] && { echo "СТОРОЖ: свободно $f ГБ -- стоп quantize"; pkill -f "rwkv_quant import quantize"; }; done ) & G=$!
( while sleep 300; do echo "  [$(date +%T)] своп $(sysctl -n vm.swapusage | awk '{print $6}') диск $(df -g ~ | awk 'NR==2{print $4}') ГБ"; done ) & S=$!
trap "kill $G $S 2>/dev/null" EXIT
sw() { sysctl -n vm.swapusage | awk '{print $6}'; }
step() { # метка, python-код
  echo "== $1 старт $(date '+%T') своп $(sw)"
  /usr/bin/time -l $PY -u -c "$2" > $W/log_$1.txt 2>&1; rc=$?
  grep -v "^emb-gather" $W/log_$1.txt | grep -E "gptq\] [0-9]+ матриц|autopick|ВСЕГО|saved|РАВНО|РАЗНЫЕ|Error|Traceback|peak memory footprint|real" | sed 's/^/  /'
  echo "== $1 конец $(date '+%T') код $rc своп $(sw)"
}
qz() { # метка, чекпоинт, пресет
  step $1 "
import time, sys; sys.path.insert(0,'.')
from rwkv_quant import quantize
t0=time.time()
quantize('$2', '$W/$1.rwkvq', preset='$3', tokenizer='$T', verbose=True)
print('ВСЕГО quantize: %.0f с' % (time.time()-t0), flush=True)"
  ls -la $W/$1.rwkvq 2>/dev/null | awk '{print "  файл", $5}'
}
# 0.4B: прежний w1014/g1d_0.4b.safetensors — из этого же чекпоинта? (сверка тензоров, без записи)
step chk_0p4b_st "
import mlx.core as mx
from mlx.utils import tree_flatten
from rwkv_metal.model.convert import load_pretrained
m,_=load_pretrained('$K/rwkv7-g1d-0.4b-20260210-ctx8192.pth'); a=dict(tree_flatten(m.parameters()))
b=mx.load('models/g1d_0.4b.safetensors')
bad=[k for k in a if k not in b or a[k].shape!=b[k].shape or not mx.array_equal(a[k],b[k]).item()]
print('ключей', len(a), len(b), 'РАВНО' if not bad and len(a)==len(b) else 'РАЗНЫЕ %s' % bad[:5])"
qz g1d_0p4b_reduction $K/rwkv7-g1d-0.4b-20260210-ctx8192.pth reduction
qz g1d_0p4b_compression $K/rwkv7-g1d-0.4b-20260210-ctx8192.pth compression
until grep -q "sha OK rwkv7-g1d-1.5b" $DL; do grep -qE "MISMATCH|EXIT [1-9]" $DL && { echo "скачивание 1.5B не удалось"; exit 4; }; sleep 30; done
step st_g1d_1p5b "
from rwkv_metal.model.convert import save_converted
save_converted('$K/rwkv7-g1d-1.5b-20260212-ctx8192.pth', '$W/g1d_1.5b.safetensors')"
qz g1d_1p5b_reduction $K/rwkv7-g1d-1.5b-20260212-ctx8192.pth reduction
qz g1d_1p5b_compression $K/rwkv7-g1d-1.5b-20260212-ctx8192.pth compression
echo "== конец $(date '+%F %T') диск $(df -g ~ | awk 'NR==2{print $4}') ГБ"
