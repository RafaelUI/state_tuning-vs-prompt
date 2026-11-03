#!/bin/bash
# 27.10 повтор: g1d 2.9B (первый hf завис на 93 %: .incomplete полного размера, но с дырами, sha не тот).
D=models; REV=f2adc5e293f0; n=rwkv7-g1d-2.9b-20260131-ctx8192.pth; h=83f91e0d04b4c430b5b945d6ee4e5ad9ef22d6cd8d7a67878337a4c1fc4f9f61
echo "== $n старт $(date '+%F %T')"
hf download BlinkDL/rwkv7-g1 "$n" --revision $REV --local-dir $D || exit 2
got=$(shasum -a 256 "$D/$n" | cut -d' ' -f1)
[ "$got" = "$h" ] && echo "sha OK $n" || { echo "sha MISMATCH $n $got"; exit 3; }
echo "конец $(date '+%F %T')"
