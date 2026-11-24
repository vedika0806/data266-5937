# HW12 — METRICS

<!-- BEGIN part1 -->
## Part 1 — GPTQ quantization (Wikitext-2)

Model: `Qwen/Qwen2.5-0.5B` @ revision `060db6499f32faf8b98477b0a26969ef7d8b9987` · Data: `Salesforce/wikitext` / `wikitext-2-raw-v1` @ `b08601e04326c79dfdd32d625aee71d232d685c3` · GPU: Tesla T4
GPTQ: 4-bit, group size 128, desc_act=False, sym=True,
calibration = 128 × 512-token windows from the **train** split (seed 5937).
Evaluation: **test** split, non-overlapping 2048-token windows (146 windows, 298,862 predicted tokens).
Latency: batch 1, 30 prompts × 128 tokens, greedy, exactly 64 new tokens, 3 warm-up requests.
Peak memory: `torch.cuda.max_memory_allocated` during the latency benchmark. Size: `save_pretrained` folder (base saved in FP16).

**Table HW11.1 — GPTQ comparison on the same LLM/task.**

| Configuration                      |   Quality metric (Wikitext-2 test PPL ↓) | p50 / p95 latency (ms)   |   Peak memory (GB) |   Size on disk (MB) |
|:-----------------------------------|-----------------------------------------:|:-------------------------|-------------------:|--------------------:|
| Base LLM (Qwen/Qwen2.5-0.5B, FP16) |                                   13.07  | 2113.3 / 4758.9          |              1.012 |               999.5 |
| GPTQ-quantized LLM (4-bit, g128)   |                                   14.354 | 5927.8 / 6357.9          |              0.944 |               471.1 |

Additional measurements:

| Configuration   |   eval loss (nats/token) | prefill 512 tok p50 / p95 (ms)   |   ms / generated token (p50) |   peak memory during PPL eval (GB) |   weights only (MB) |
|:----------------|-------------------------:|:---------------------------------|-----------------------------:|-----------------------------------:|--------------------:|
| Base FP16       |                   2.5703 | 39.78 / 51.20                    |                        33.02 |                              4.119 |               988.1 |
| GPTQ 4-bit      |                   2.6641 | 90.79 / 126.39                   |                        92.62 |                              4.039 |               459.4 |

![base vs gptq](p1_base_vs_gptq.png)
![latency distribution](p1_latency_distribution.png)

Sample greedy continuation of "The history of the Roman Empire":
- Base: 'The history of the Roman Empire is one of the most fascinating in the world. It is a fascinating history because it is the history of the world. It is a history of the world because it is the history of the world.'
- GPTQ: 'The history of the Roman Empire is one of the most fascinating and complex stories in the history of the world. The Romans were a people who had a unique blend of culture, history, and philosophy. They were also known for their'

### Part 1 findings

- **Quality cost:** perplexity rose from 13.070 to 14.354 (+9.8%), i.e. eval loss 2.5703 → 2.6641 nats/token.
- **Size:** 1000 MB → 471 MB on disk (2.12× smaller). It is not 4× because GPTQ quantizes only the transformer
  linear layers; the Qwen/Qwen2.5-0.5B embedding matrix (151,936 × 896, tied with the LM head) stays in FP16, and each group of 128 weights also stores a scale and zero point.
- **Peak GPU memory during generation:** 1.012 GB → 0.944 GB (1.07× lower). KV cache and activations stay FP16, so memory shrinks less than the weights.
- **Latency (batch 1, 128-token prompt, 64 new tokens):** p50 2113.3 → 5927.8 ms, p95 4758.9 → 6357.9 ms
  — the GPTQ model is 2.81× slower. Prefill of 512 tokens: 39.78 → 90.79 ms.
  Kernel used: n/a. For a 0.5 B model at batch 1, each decoding step is dominated by per-layer kernel-launch overhead rather than by reading weights, and the 4-bit weights must be dequantized inside the kernel, so GPTQ gives no real speed-up here; its gain is memory and storage.
- **Quantization cost:** 1.5 min on Tesla T4 with 128 calibration windows — no training and no labels.
<!-- END part1 -->
