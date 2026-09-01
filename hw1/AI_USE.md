# AI_USE.md

## Which parts did you use an assistant for, and which did you write yourself?

I used AI assistance to figure out how to fulfill the standing requirements as instructed in the assignment such that there is a seamless incremental commit history and git operations.
I also used AI to better my code for pytorch and tensorflow and to learn CUDA C programming from scratch. I used coding assistance for CUDA through AI models as I am still in learning phase.

The rest of the work was my own and I wrote the error analysis myself, filling in the measured numbers into result tables, analyzing them and making conclusion.

## Give one specific thing it produced that was wrong?

While I was working with CUDA programming, I observed that the values for N=256 matrix were strange and checked for it. I realized that it did not have the warmup kernel launch (that was also mentioned in the standing instructions) before the timed measurements.
The results initially indicated that the CPU was faster than the GPU for small matrix size, which was contradicted in N=1024 and N=4096. 

## How did you find out? What did the failure look like?

I realized that when the GPU kernel time for N=1024 was 1.054 ms but for N=256 was 91.8ms even though the N=1024 had more floating point operations.So there was a measurement inaccuracy rather than a GPU problem.

## What did you change, and why does your version work?

I identified that the very first CUDA kernel launch in a process pays a one-time cost for CUDA context creation and kernel JIT compilation and this cost was captured in my timed measurement. I added a warn-up kernel immediately before the timed-kernel so that one time initializtion cost is absorbed before the initialization begins.The result became consistent with the rest of the data and the GPU was faster than the CPU at any expected times.

