[English](README.md) · [Русский](README.ru.md)

# Models

Put the models here (they are not in git). Checksums — `SHA256SUMS` (`shasum -a 256 -c SHA256SUMS`).

| file | where from |
|---|---|
| `rwkv_vocab_v20230424.txt` | the RWKV World vocabulary ([BlinkDL/ChatRWKV](https://github.com/BlinkDL/ChatRWKV), `rwkv_pip_package/src/rwkv/rwkv_vocab_v20230424.txt`) |
| `rwkv7-g1d-0.4b-20260210-ctx8192.pth`, `rwkv7-g1d-1.5b-20260212-ctx8192.pth`, `rwkv7-g1d-2.9b-20260131-ctx8192.pth` | [BlinkDL/rwkv7-g1](https://huggingface.co/BlinkDL/rwkv7-g1), revision `f2adc5e293f0` (g1d has since been removed from the current revision) — `experiments/1027_matrix/dl_g1d.sh` |
| `g1d_0.4b.safetensors`, `g1d_1.5b.safetensors` | `.pth` → bf16 `.safetensors` with the rwkv-metal converter (`save_converted`) — `experiments/1027_matrix/build_g1d.sh` |
| `g1d_{0p4b,1p5b}_{reduction,compression}.rwkvq` | [rwkv-quant](https://github.com/RafaelUI/rwkv-quant) `d228746`, `quantize()` defaults — `build_g1d.sh` |
| `g1j_{1p5b,2p9b}_compression.rwkvq` | g1j 1.5B / 2.9B (`rwkv7-g1j-*-20260831-ctx16384.pth`), COMPRESSION from [the quantization article](https://github.com/RafaelUI/rwkv-quant_ARTICLE) |

`build_g1d.sh` writes its results to `experiments/1027_matrix/` — move them to `models/`.

REDUCTION is ~2.1× smaller than bf16 and nearly lossless; COMPRESSION is ~3.1× smaller (4-bit projections). The presets are
described in [Quantizing RWKV-7](https://github.com/RafaelUI/rwkv-quant_ARTICLE) and implemented in [rwkv-quant](https://github.com/RafaelUI/rwkv-quant).
