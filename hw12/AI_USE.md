# AI_USE.md

## 1. What I used AI for vs. wrote myself
- **AI ** helped in the GPTQ quantization and Wikitext-2 evaluation/benchmark code, the knowledge-distillation loss and training loop, the temperature and alpha sweeps, the teacher-vs-student benchmark, the per-class analysis,
  the figures.
  When my runs failed, it diagnosed the errors and gave me fixed cells (PyTorch/TorchAudio CUDA mismatch, bf16 on the T4, the Marlin
  GPTQ kernel not compiling on the T4, failed git pushes, peak memory measured with both models on the GPU).
- **Me:** made the setup decisions (base LLM, A100 → T4 when Colab had no A100/L4, 1 epoch per student, training every student on the same
  seeded 30k subset to fit the T4 time budget, retraining a fresh alpha = 0.5 student instead of reusing the temperature-sweep run, loading the
  teacher from my local download), applied the fixes, and recorded the findings.

## 2. One specific wrong thing the AI produced?
My pushes to GitHub were failing because the 400 MB teacher archive had been saved into hw12/. I had to fix the same since the subsequent results were not getting recorded and failing without any error.

## 3. How I found out it was wrong
The cell only needed to remove the teacher archive, but the "moved out of repo" line in its output and in RUN_LOG.txt listed
hw9/latent_unet.pt as well. Checking the commit (git show --stat e353a5e) confirmed that the push had deleted the 59 MB HW9 model file
from GitHub (59053509 -> 0 bytes), a file that belongs to another assignment. 

## 4. What I changed and why the fix works
I restored the file from git history and pushed it again:

    git checkout 148a500 -- hw9/latent_unet.pt     (148a500 = last commit that still contained it)
    git commit -m "hw9: restore latent_unet.pt (accidentally removed by the hw12 large-file cleanup)"   -> commit 42cb819
