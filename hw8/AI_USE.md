# AI_USE.md — HW8 Mini Diffusion Model (DDPM → DDIM)

## 1. What I used AI for vs. what I did myself

**AI was used for:**
- Initial understanding and architecture design methods: forward diffusion `q_sample`, the MiniUNet noise
  predictor (sinusoidal timestep embedding + residual blocks).
- After I showed it my bad samples: diagnosing the cause, and code for the diagnostic cells
  (real noised x_T, moment-matched x_T) and the scaled-schedule ablation.

**What I did myself:**
-After checking the bad samples, I observed that DDPM gave smooth scribbles instead of digits, DDIM gave speckled, noisy output and DDIM looked the same at 5 steps as at 100.

## 2. One specific wrong thing the AI produced

-Quality. DDPM (100 steps) and DDIM (50 steps) give comparable, clean digits. DDIM (20 steps)
is still recognizable but may show slightly thicker or blurrier strokes or occasional malformed
digits, because each step takes a larger jump along the trajectory.
-The AI made me  assume that the forward process ends at pure noise (ᾱ_T ≈ 0), which is true for the DDPM paper's T = 1000, but with T = 100 and
β linear 1e-4 → 0.02, ᾱ_T = 0.3636, so about 60% of the image is still there at t = T.

## 3. How I found out it was wrong
My actual results contradicted the prediction:
- d_sampler_comparison.png: DDPM gave smooth strokes but mostly scribbles, not real digits.
  DDIM-50 and DDIM-20 were clearly worse than DDPM, speckled and noisy, not "comparable, clean digits."
- d_ddim_step_sweep.png: DDIM at 100, 50, 20, 10 and 5 steps looked almost identical, even at 5 steps.
- c_ddpm_trajectory.png: digit shapes were already visible at t = 80, too early for pure noise.

To confirm the cause, I compared the real x_T distribution with the N(0, I).

## 4. What I changed and why the fix works

The assigned config (T = 100, β 1e-4 → 0.02) stays as the main deliverable. I added two fixes:

1. **Moment-matched x_T (no retraining).** Sample
   x_T ~ N(√ᾱ_T · μ_pix, ᾱ_T · σ²_pix + 1 − ᾱ_T) from per-pixel MNIST statistics. This matches the
   mean and variance of the true q(x_T), so the starting point is closer to what the model was
   trained on. DDPM went from scribbles to mostly recognizable digits. DDIM improved only partly,
   because a per-pixel Gaussian ignores pixel correlations: DDPM's per-step noise can correct the
   remaining mismatch, but deterministic DDIM carries it through.

2. **Scaled schedule ablation (retrained 20 epochs).** β 1e-3 → 0.2 (the assigned β × 1000/T), which
   gives ᾱ_T ≈ 0, so N(0, I) really is the correct starting distribution. All three samplers then
   produced clean digits, and DDIM-50/20 reached near-DDPM quality at about 2× and 5× speed-up.
   This shows the speckle came from the start mismatch, not from my DDIM implementation.

