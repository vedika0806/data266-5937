# AI_USE.md — HW2.5

## Which parts did you use an assistant for, and which did you write yourself?

I used AI to understand the mechanics of GPU, its architecture and working. I required help to construct the code to run it effectively so that I could understand precision, bandwidth and the cost of attention.
I diagnosed anomalies like the FP32 N=1024 bimodality and the L=4096 attention timing dip. I wrote the final interpretation in METRICS.md.

## Give one specific thing it produced that was wrong. Paste it.

It assumed an incorrect theoretical peak value: An early peak_tflops table set BF16 peak to 165.2 and FP16 peak to 330.3, comparing
them under inconsistent accumulation-mode assumptions.

## How did you find out?

The achieved TFLOPS calculation was correct: TFLOPS = (2 × N³) / elapsed time / 10¹². The mistake was in calculating the percentage of theoretical peak for FP16. AI initially prompted me to use 330.3 TFLOPS, which represents FP16 accumulation, but the benchmark used the FP32-accumulation convention. Therefore, FP16 should be compared against the 165.2 TFLOPS theoretical peak, the same peak used for BF16. 
For example, an achieved FP16 performance of 169.37 TFLOPS was incorrectly reported as 51.3% using 330.3 TFLOPS, but the corrected value is 103.5% using 165.2 TFLOPS.


## What did you change, and why does your version work?

I corrected the theoretical FP16 peak from 330.3 TFLOPS to 165.2 TFLOPS because the benchmark used FP32 accumulation, making the FP16 comparison consistent with BF16. The revised version works because it measures completed GPU operations rather than asynchronous CPU launch time and compares each measured result against the appropriate theoretical peak for the same accumulation mode.


