[English](README.md) · [Русский](README.ru.md)

# Stages

One folder per stage; the name starts with the date (MMDD). Each holds the run scripts and summaries, the summaries
(`*_summary.md`, in Russian), the stage's input data and `outputs.tar.gz` with raw answers, teacher caches and training logs.
How to run — [REPRODUCE.md](../REPRODUCE.md).

| folder | section of the write-up ([docs/results.md](../docs/results.md)) |
|---|---|
| `1019_embed_fold`, `1020_embed_cut`, `1020_embed_bank` | Embedder: query instruction (historical scripts, `embed-run`) |
| `1020_neuron_0p4b` | Generation: "Neuron" (0.4B, protocols c1–c4, form transfer) |
| `1022_neuron_1p5b` | Generation: 1.5B (23 Oct) |
| `1027_matrix` | Scale × quantization; early stopping |
| `1029_prompts` | Five different prompts |
| `1030_provocations_control` | Identity provocations; Strong control (`src/` — the TopLayerFinetune code before it was committed to SwiftRWKV) |
| `1031_prov_training` | Provocations in training |
| `1101_dialogues` | Role over time (`gate/` — binary checks) |
| `1101_classification` | Instruction following: classification 0.4B / 1.5B, data-size curve |
| `1102_audit` | Reliability of the grading |
| `1103_cost` | Cost |
