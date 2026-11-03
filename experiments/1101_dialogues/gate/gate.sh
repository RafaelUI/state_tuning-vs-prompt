#!/bin/bash
# 01.11: гейт gen-run с --dialogues (SwiftRWKV рабочее дерево поверх 148861d).
# (1) старые плечи матрицы (0.4B: без промпта, промпт, c4 сид 7) — ответы побитно равны прежним (w1027/m);
# (2) диалоги из одного хода (вопросы questions.txt) — ход 0 побитно = одиночный вопрос того же плеча;
# (3) Scripts/check_genrun_cli.sh — свойства ключей --distill.
set -e
G=${SWIFTRWKV:-../SwiftRWKV}/.build/release/gen-run; W=experiments/1101_dialogues/gate; N=data/neuron
M=models/g1d_0.4b.safetensors; V=models/rwkv_vocab_v20230424.txt; O=experiments/1027_matrix/m/g1d_0p4b_bf16
SYS=$(cat $N/system.txt)
$G --model $M --vocab $V --questions $N/questions.txt --system "$SYS" --mode none --max-tokens 120 --samples 10 --temp 1.0 --top-p 0.6 --seed 100 --out $W/id_none_s10.json > $W/id_none.log 2>&1
$G --model $M --vocab $V --questions $N/checks_q.txt --system "$SYS" --mode none --state-in $O/c4_s7.safetensors --max-tokens 80 --samples 4 --temp 1.0 --top-p 0.6 --seed 200 --out $W/v2_c4_s7_s4.json > $W/v2.log 2>&1
$G --model $M --vocab $V --questions $N/questions.txt --system "$SYS" --mode inline --max-tokens 120 --samples 10 --temp 1.0 --top-p 0.6 --seed 100 --out $W/id_inline_s10.json > $W/id_inline.log 2>&1
/usr/bin/python3 -c "
import json
qs=[l.strip() for l in open('$N/questions.txt',encoding='utf-8') if l.strip() and not l.startswith('#')]
json.dump([{'id':f'q{i}','turns':[q]} for i,q in enumerate(qs)], open('$W/one_turn.json','w'), ensure_ascii=False)"
$G --model $M --vocab $V --dialogues $W/one_turn.json --system "$SYS" --mode none --max-tokens 120 --samples 10 --temp 1.0 --top-p 0.6 --seed 100 --out $W/d_none.json > $W/d_none.log 2>&1
$G --model $M --vocab $V --dialogues $W/one_turn.json --system "$SYS" --mode inline --max-tokens 120 --samples 10 --temp 1.0 --top-p 0.6 --seed 100 --out $W/d_inline.json > $W/d_inline.log 2>&1
$G --model $M --vocab $V --dialogues $W/one_turn.json --system "$SYS" --mode none --state-in $O/c4_s7.safetensors --max-tokens 120 --samples 10 --temp 1.0 --top-p 0.6 --seed 100 --out $W/d_c4_s7.json > $W/d_c4.log 2>&1
/usr/bin/python3 - <<'PY'
import json
W='experiments/1101_dialogues/gate'; O='experiments/1027_matrix/m/g1d_0p4b_bf16'
ok=True
for f in ['id_none_s10.json','v2_c4_s7_s4.json','id_inline_s10.json']:
    a=json.load(open(f'{W}/{f}'))['rows']; b=json.load(open(f'{O}/{f}'))['rows']
    e=a==b; ok&=e; print('старое плечо', f, 'РАВНО' if e else 'РАЗНЫЕ', sum(x!=y for x,y in zip(a,b)), 'из', len(a))
for d,ref in [('d_none.json',f'{O}/id_none_s10.json'),('d_inline.json',f'{O}/id_inline_s10.json'),('d_c4_s7.json',f'{O}/id_c4_s7_s10.json')]:
    D=json.load(open(f'{W}/{d}'))['dialogues']; R=json.load(open(ref))['rows']
    e=[[s[0] for s in x['answers']] for x in D]==[r['answers'] for r in R] and [x['turns'][0] for x in D]==[r['question'] for r in R]
    ok&=e; print('ход 0 диалога', d, '= одиночный вопрос' if e else 'РАСХОЖДЕНИЕ')
print('ГЕЙТ', 'ПРОЙДЕН' if ok else 'НЕ ПРОЙДЕН'); raise SystemExit(0 if ok else 1)
PY
cd ${SWIFTRWKV:-../SwiftRWKV} && ./Scripts/check_genrun_cli.sh
