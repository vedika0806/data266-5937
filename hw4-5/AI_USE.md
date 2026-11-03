# AI_USE.md

## 1. Which parts I used AI for vs. wrote myself?

I used AI throughout in HW4.5. It also helped me debug the lab environment: a
PyTorch/MKL threading conflict, a missing C compiler for Triton, a vLLM multiprocessing crash, and FlashInfer needing
`nvcc`.
I worked on HW 4.a by myself for the BERT fine-tune and batch/technique sweeps, `scripts/serve_bench.py` 

## 2. One specific wrong thing the AI produced?

The batch-size sweep reported a maximum baseline batch of 1024 (and 1024 for bf16, 2048 for gradient checkpointing)
on a 24 GB RTX 4090. That number was wrong, and so was the Part B comparison built on it: the notebook used the
largest "successful" batch (1024) as the reference batch.

## 3. How I found out it was wrong?

The sweep slowed down dramatically. Batch 256 finished in about 28 s, but batch 512 took 553 s and batch 1024 took
781 s (bf16 batch 1024 took 949 s), and the cell looked stuck. The logged peak memory for those runs was 43 GB and
86 GB reserved, which is more than the card physically has, and throughput collapsed from 552 to about 20 samples/s.


## 4. What I changed and why the fix works?

I backed up all existing results. I  made changes and then ran the remaining jobs with a capped copy of the scripts that calls `torch.cuda.set_per_process_memory_fraction(0.95)`, and flagged every earlier
sweep point whose peak reserved memory exceeded 95% of VRAM (`runs/spill_flags.csv`). With the cap, PyTorch's caching
allocator raises `OutOfMemoryError` before the driver can oversubscribe VRAM, so oversized batches fail cleanly in
seconds instead of crawling through system RAM. I re-derived the true limits (baseline 256, bf16 256, gradient
checkpointing 1024, 8-bit optimizer 256), set the Part B reference batch to 256, re-ran the one gradient-accumulation
benchmark needed at that batch.
