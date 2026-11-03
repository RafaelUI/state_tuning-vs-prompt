#!/bin/bash
# 27.10: g1d 1.5B и 2.9B из истории BlinkDL/rwkv7-g1 (ревизия f2adc5e293f0 — до удаления 20.04), проверка sha256 по LFS.
set -e
D=models; REV=f2adc5e293f0
for f in rwkv7-g1d-1.5b-20260212-ctx8192.pth:e30cf1c800d1ddd86fe65f79d53454147fefe1d736a2eb3441e1a13ff1f6627e rwkv7-g1d-2.9b-20260131-ctx8192.pth:83f91e0d04b4c430b5b945d6ee4e5ad9ef22d6cd8d7a67878337a4c1fc4f9f61; do
  n=${f%%:*}; h=${f##*:}
  echo "== $n старт $(date '+%F %T')"
  hf download BlinkDL/rwkv7-g1 "$n" --revision $REV --local-dir $D
  got=$(shasum -a 256 "$D/$n" | cut -d' ' -f1)
  [ "$got" = "$h" ] && echo "sha OK $n" || { echo "sha MISMATCH $n $got"; exit 3; }
done
echo "конец $(date '+%F %T')"
