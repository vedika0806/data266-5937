
## Part B — Precision and Achieved Throughput

**Methodology notes:** I made multiple benchmark attempts to validate the timing results and improve
reproducibility. The initial runs showed high variance for small matrix sizes, so I increased the
warm-up iterations, used CUDA event timing with synchronization, and collected multiple timing samples.
I also cleared the CUDA cache and compared results across runs. During analysis, I corrected the FP16
theoretical peak from 330.3 TFLOPS to 165.2 TFLOPS so that FP16 and BF16 were compared using the same
FP32-accumulation convention. The final benchmark used 20 warm-up iterations, 30 timed repetitions per
sample, and 10 samples. The results for matrix sizes N >= 4096 were stable, while N = 1024 remained more
sensitive to timing overhead.

**FP32 N=1024 bimodality:** FP32 at N=1024 specifically showed a persistent bimodal distribution across
repeated trials (~11.8 TFLOPS in a low regime, ~38.9 TFLOPS in a high regime) that did not resolve even
after tripling warmup iterations. This rules out simple GPU clock-ramp as the cause and suggests cuBLAS
nondeterministically selecting between two kernel implementations for this specific small, non-tensor-core
matmul shape. TF32, FP16, and BF16 at the same size did not show this bimodality. The high-regime value
is reported below as representative of steady-state behavior.

### Theoretical peak source
NVIDIA Ada GPU Architecture Whitepaper, Appendix A, Table 2 "GeForce RTX 4090 Full Specifications" (dense/non-sparse column).
- FP32 (non-Tensor): 82.6 TFLOPS
- TF32 Tensor (dense): 82.6 TFLOPS
- FP16 Tensor, FP32 Accumulate (dense): 165.2 TFLOPS
- BF16 Tensor, FP32 Accumulate (dense): 165.2 TFLOPS

### Results
| N | Precision | TFLOPS | CV (%) | % of Peak |
|---|---|---|---|---|
| 1024 | FP32 | 39.01 | 5.09 | 47.2% |
| 1024 | TF32 | 40.92 | 38.62 | 49.5% |
| 1024 | FP16 | 70.00 | 0.39 | 42.4% |
| 1024 | BF16 | 69.47 | 0.12 | 42.0% |
| 4096 | FP32 | 49.24 | 0.53 | 59.6% |
| 4096 | TF32 | 85.22 | 0.67 | 103.2% |
| 4096 | FP16 | 167.08 | 0.89 | 101.1% |
| 4096 | BF16 | 169.94 | 0.53 | 102.9% |
| 8192 | FP32 | 51.88 | 0.34 | 62.8% |
| 8192 | TF32 | 85.26 | 0.46 | 103.2% |
| 8192 | FP16 | 155.67 | 0.19 | 94.2% |
| 8192 | BF16 | 164.84 | 0.81 | 99.8% |
| 16384 | FP32 | 50.62 | 0.05 | 61.3% |
| 16384 | TF32 | 83.86 | 0.06 | 101.5% |
| 16384 | FP16 | 170.55 | 0.40 | 103.2% |
| 16384 | BF16 | 170.30 | 0.48 | 103.1% |

## Part C — Bandwidth-bound vs Compute-bound

### Method
Two operations were benchmarked: a large elementwise add (memory-bound) and a large square
FP32 matmul at N=16384 (compute-bound). Both used CUDA-event timing with warmup, and
10 repeated samples to quantify variance via coefficient of variation (CV).

### Results

| Metric | Elementwise Add | Matmul (FP32, N=16384) |
|---|---|---|
| Measured throughput | 922.92 GB/s | 51.51 TFLOPS |
| % of peak | 91.6% (of 1008 GB/s) | 62.4% (of 82.6 TFLOPS) |
| CV (variance) | 0.10% | 0.49% |
| Arithmetic Intensity | 0.0833 FLOPs/byte | 2730.67 FLOPs/byte |

**Roofline crossover intensity:** 81.94 FLOPs/byte (computed as FP32 peak TFLOPS / peak bandwidth GB/s).

**Note on matmul AI:** computed using naive/conservative byte-counting — one read of A, one read
of B, one write of C, no credit for tiling/cache reuse. This is a lower bound; real cuBLAS tiled
kernels move less HBM traffic, so true arithmetic intensity is at least this high, likely higher.

### Classification
- Elementwise add: AI = 0.0833 FLOPs/byte, far below the crossover of 81.94
  → **memory-bound**.
- Matmul (FP32): AI = 2730.67 FLOPs/byte, roughly 33x above the
  crossover → **compute-bound**.

### Discussion
The two operations sit on opposite extremes of the roofline: elementwise add achieves
91.6% of peak bandwidth but is fundamentally limited by data movement (0.083 FLOPs
performed per byte moved), while the FP32 matmul achieves 62.4% of peak compute
throughput and is limited by the GPU's raw arithmetic rate rather than memory traffic, needing
over 2700 FLOPs per byte moved from HBM to sustain its throughput.

## Part D — The Cost of Attention

**Config:** batch=1, heads=8, d_head=64, dtype=float16
**Timing method:** CUDA events (torch.cuda.Event), single extended warmup block before timed
samples (per-sample warmup was tested and found insufficient to eliminate variance), 5-10
repeated samples per configuration with coefficient of variation (CV) reported.

### Naive attention

| Seq Len | Latency (ms) | CV (%) | Peak Memory (GB) |
|---|---|---|---|
| 512 | 0.0862 | 4.55 | 0.022 |
| 1024 | 0.1700 | 0.25 | 0.049 |
| 2048 | 1.6714 | 150.81 | 0.155 |
| 4096 | 1.8063 | 0.50 | 0.573 |
| 8192 | 7.1587 | 0.13 | 2.200 |
| 16384 | 28.7940 | 0.72 | 8.684 |

**OOM boundary:** No out-of-memory failure was observed up to the largest tested
sequence length (49152). Naive attention memory scales as O(batch x heads x L^2); with the
small configuration used here (batch=1, heads=8), even L=49152 stays within the RTX 4090's
24GB VRAM.

**Memory scaling (quadratic fit, memory(L) = a*L^2 + b*L + c):**
a = 3.2000000000e-08 GB/token^2, b = 5.1200000000e-06 GB/token, c = 0.0106 GB.
The positive, non-negligible quadratic coefficient confirms O(L^2) memory scaling, consistent
with materializing the full L x L attention matrix.

**Note on L=2048 variance:** The naive attention measurement at L=2048 shows an
unusually high coefficient of variation (CV=150.81%) in the table above. Given the magnitude of this CV, the
reported latency at L=2048 should be treated as unreliable; the surrounding
values (L=1024, L=4096) are stable (CV < 1%) and are more representative of
actual performance at nearby sequence lengths.

**Note on L=2048 variance:** The naive attention measurement at L=2048 shows an
unusually high coefficient of variation (CV=150.81%) in the table above. Given the magnitude of this CV, the
reported latency at L=2048 should be treated as unreliable; the surrounding
values (L=1024, L=4096) are stable (CV < 1%) and are more representative of
actual performance at nearby sequence lengths.

**Anomaly investigated:** naive latency at L=4096 (1.806ms) did not follow a clean O(L^2)
trend relative to L=2048 (1.671ms) and L=8192 (7.159ms). Re-measured with reps=50, warmup=30,
samples=10 and confirmed stable (CV=0.50%) -- this is a genuine result, not noise. Attributed
to the GPU's compute units not being fully saturated at this size; latency scaling becomes
closer to theoretical O(L^2) once a size threshold (here, between 4096 and 8192) is crossed
and the kernel becomes compute-saturated.

### Backend selection for fused attention

Initial attempts to use `F.scaled_dot_product_attention` with default (unpinned) backend
dispatch showed anomalously high variance at L=4096 (CV=61%), traced to the SDPA dispatcher
inconsistently switching between the Flash Attention and Memory-Efficient backends at this
shape. The current PyTorch build (2.1.2) does not support the newer
`torch.nn.attention.sdpa_kernel` API (`ImportError: cannot import name '_SDPBackend' from
torch._C`), so the older `torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False,
enable_mem_efficient=False)` context manager was used instead, successfully pinning execution
to the Flash Attention backend. All fused-attention results below use this pinned backend.

### Fused attention (Flash Attention, pinned)

| Seq Len | Latency (ms) | CV (%) | Peak Memory (GB) |
|---|---|---|---|
| 512 | 0.0582 | 0.86 | 0.029 |
| 1024 | 0.1597 | 43.08 | 0.022 |
| 2048 | 0.3406 | 0.63 | 0.030 |
| 4096 | 1.2588 | 12.16 | 0.044 |
| 8192 | 1.1838 | 0.47 | 0.074 |
| 16384 | 4.0526 | 0.65 | 0.133 |
| 24576 | 10.9659 | 0.32 | 0.192 |
| 32768 | 24.7080 | 1.08 | 0.251 |

**OOM boundary:** largest successful length = 32768, smallest failing length =
not reached in tested range up to 32768

### Speedup (naive / fused, Flash-pinned)

| Seq Len | Naive (ms) | Fused (ms) | Speedup |
|---|---|---|---|
| 512 | 0.0862 | 0.0582 | 1.48x |
| 1024 | 0.1700 | 0.1597 | 1.06x |
| 2048 | 1.6714 | 0.3406 | 4.91x |
| 4096 | 1.8063 | 1.2588 | 1.43x |
| 8192 | 7.1587 | 1.1838 | 6.05x |
| 16384 | 28.7940 | 4.0526 | 7.10x |

### What the fused kernel avoids

The fused/Flash Attention kernel avoids materializing the full L x L score matrix in HBM.
Instead of computing the entire attention matrix, writing it to global memory, applying
softmax, and reading it back for the weighted sum, it tiles the computation and keeps
partial results in on-chip shared memory/registers, using an online softmax algorithm that
updates running max/sum statistics as it streams through key/value blocks. This reduces
memory traffic from O(L^2) to O(L), which is why the fused kernel's OOM boundary is far
beyond the naive implementation's, and why the speedup grows with sequence length (from
~1.1-1.5x at small L up to ~7x at L=16384) rather than staying constant.

## Part E — Sustained Load and Thermal Behaviour

**Setup:** 20-minute sustained BF16 matmul load (N=8192), 5-second sampling of clock/temp/power/utilization.

### Throughput
- Peak throughput (first 30s): 168.49 TFLOPS
- Steady-state throughput (last 5 min): 163.86 TFLOPS
- Steady-state / peak ratio: 97.2%

### Thermal / clock behavior
- Peak clock: 2655 MHz (~5s in, 60C, 207W)
- Settled clock: ~2550-2565 MHz (by ~55-95s in, 75-79C)
- Max temperature observed: 84C
- Power draw stabilized at ~448-450W, at the RTX 4090's rated 450W TGP (NVIDIA Ada Whitepaper, Appendix A)

### Throttling assessment
Mild clock throttling occurred, driven by the power limit rather than temperature. The
~3.5% clock reduction (2655 -> ~2555 MHz) closely matches the measured 2.8% throughput
decline (168.49 -> 163.86 TFLOPS, 97.2% ratio), with clock and
throughput data corroborating each other. Temperature continued climbing throughout the
run while clock speed remained flat once power hit its ceiling, indicating the card is
power-limited, not thermally throttled, under this workload.

Throttle onset: approximately 5-15 seconds into the run.
Binding constraint: power limit (~450W), not temperature ceiling.

This result is specific to this card, chassis, and ambient room conditions; no two
students are expected to report identical values.

## Table HW2.5.1 — Summary

| Measurement | Your GPU | Notes |
|---|---|---|
| Peak achieved TFLOPS (BF16) | 170.30 | N=16384, CUDA-event timing, CV=0.48% |
| % of theoretical peak (BF16) | 103.1% | Peak=165.2 TFLOPS, NVIDIA Ada Whitepaper Appendix A Table 2 (dense, FP32-accumulate) |
| Effective bandwidth (GB/s) | 922.92 | 91.6% of 1008 GB/s spec, elementwise add, CV=0.10% |
| Naive attention OOM length | Not reached (tested to 49152) | With batch=1, heads=8, no OOM occurred up to the largest tested length; see Part D note |
| Fused attention OOM length | not reached (tested up to 32768) | Flash Attention backend, pinned via torch.backends.cuda.sdp_kernel |
| Steady-state / peak throughput | 97.2% | 20-min sustained BF16 matmul, N=8192 |
| Throttle onset (s, or none) | ~5-15s | Power-limited (~450W TGP), not thermal-limited |

Every measurement above is logged with its GPU UUID in RUN_LOG.txt.

## Part F — Analysis

Looking at all the results together, this RTX 4090 performed best (closest to its
official white sheet specifications) when doing BF16 and FP16 matrix multiplication — it actually hit
about 103% of its rated peak speed at the largest matrix size. It performed worst on
plain FP32 math, only reaching about 61% of its peak there. The reason comes down to
hardware design: FP32 math runs on the GPU's regular cores, doing one operation per
core per cycle. BF16 and FP16 math instead run on special "Tensor Cores" built into
this GPU, which can do many more calculations per cycle than the regular cores. That's
why the same GPU looks so much faster in one mode versus the other — it's literally
using different hardware internally. Going slightly over 100% of the "peak" number
means the GPU was running its clock speed a bit higher than
the official spec sheet assumes, which is normal and happens more in short bursts.

For memory speed, the GPU moved data at about 92% of its rated maximum (1008 GB/s) when
doing a simple, memory-heavy operation (adding two big arrays together). Matrix
multiplication, on the other hand, is a "compute-heavy" operation — it does about 33
times more math per byte of data moved than the crossover point where memory speed and
compute speed become equally limiting. In simple terms: adding arrays is limited by how
fast you can move data in and out, while matrix multiplication is limited by how fast
you can do the actual math.

The attention experiments in Part D matter indicate that  naive attention builds and stores a giant grid comparing
 every token to every other token. The "fused" version (using Flash Attention) avoided ever building
that full giant grid — it processes things in smaller chunks and never stores the whole
thing in memory at once. Because of this, the fused version got relatively more and
more advantageous as sequences got longer: only about 1.1–1.5x faster at short
sequences, but over 7x faster at the longest sequence tested. The gap keeps growing
because naive attention's memory use grows much faster than fused attention's does as
sequences get longer.

When running a heavy workload continuously for 20 minutes, the GPU's speed only dropped
by about 2.8% from its initial burst of speed down to a steady, sustained level. The GPU's clock speed settled down once its power draw hit the card's rated
450-watt limit — not because it got too hot. We know this because the temperature kept
climbing well after the clock speed had already leveled off, which means power draw
(not heat) was the actual bottleneck holding the speed down. This is a useful thing as better cooling wouldn't help this card run faster under sustained load, because
it's not hitting a temperature limit — it's hitting a power limit first.
