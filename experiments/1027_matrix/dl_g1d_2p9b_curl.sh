#!/bin/bash
# 27.10: hf (xet) дважды зависал на 2.9B с дырами в .incomplete. Качаем curl'ом ПОВЕРХ того же .incomplete (диск не растёт), sha, mv на место.
D=models; n=rwkv7-g1d-2.9b-20260131-ctx8192.pth; h=83f91e0d04b4c430b5b945d6ee4e5ad9ef22d6cd8d7a67878337a4c1fc4f9f61
P=$D/.cache/huggingface/download/Bhu-9aNrvvSw29We4nuqzaFbunE=.$h.incomplete
echo "== curl $n старт $(date '+%F %T')"
curl -L -f -sS --retry 5 --retry-delay 10 -o "$P" "https://huggingface.co/BlinkDL/rwkv7-g1/resolve/f2adc5e293f0/$n" || exit 2
got=$(shasum -a 256 "$P" | cut -d' ' -f1)
[ "$got" = "$h" ] || { echo "sha MISMATCH $got"; exit 3; }
mv "$P" "$D/$n" && echo "sha OK $n -> $D/$n"
echo "конец $(date '+%F %T')"
