# METRICS.md — Mini Diffusion Model (DDPM → DDIM)

## Setup

| Item | Value |
|---|---|
| Dataset | MNIST train (60,000), scaled to [-1, 1] |
| Diffusion steps | T = 100 |
| Noise schedule | β linear 0.0001 → 0.02 (ᾱ_T = 0.3636) |
| Model | MiniUNet, 1.67M params, 19 conv layers, SiLU, sinusoidal timestep embedding, output 1×28×28 |
| Optimizer | Adam, lr = 0.0002 |
| Batch / Epochs | 128 / 50 |
| Seed | 5937 (SID4) |
| Device | cuda |

## (a) Forward diffusion

x_t = √ᾱ_t · x_0 + √(1 − ᾱ_t) · ε, visualized at t ∈ {0, 10, 25, 50, 75, 99} (a_forward_diffusion.png).
The digit is still visible at t = 99, which is the first sign of the terminal-SNR issue discussed below.

## (b) Training

| Metric | Value |
|---|---|
| Epoch 1 mean loss | 0.1325 |
| Final epoch mean loss | 0.0525 |
| Training time | 35.4 min |

Loss curve: b_training_loss.png. Per-epoch losses: train_loss.csv.

## (c)/(d) Sampler comparison — assigned schedule, x_T ~ N(0, I)

| Sampler | Network evals | Time for 64 imgs (s) | Speed-up vs DDPM | MSE vs DDIM-100 |
|---|---|---|---|---|
| DDPM    | 100 | 1.559   | 1.0x | n/a (stochastic) |
| DDIM-50 | 50  | 0.863 | 1.8x | 0.0002 |
| DDIM-20 | 20  | 0.304 | 5.1x | 0.0007 |

- DDIM (η = 0) deterministic for a fixed x_T: **False**
- DDPM run-to-run mean abs pixel difference (same x_T): **0.3138**

Figures: c_ddpm_samples.png, c_ddpm_trajectory.png, d_sampler_comparison.png, d_ddim_step_sweep.png.

### Observations
- **Quality.** DDPM produces smooth, clean strokes, but most samples are scribbles rather than valid
  digits. DDIM-50 and DDIM-20 are visibly worse: speckled, noisy strokes.
- **DDIM consistency.** DDIM at 100/50/20/10/5 steps from the same x_T gives nearly identical images,
  showing the deterministic x_T → x_0 mapping. Because ᾱ only spans 0.36 → 1, the
  trajectory is short and even 5 steps stays close.
- **Speed.** Cost scales with network evaluations: DDIM-50 ≈ 1.8× and
  DDIM-20 ≈ 5.1× faster than DDPM, using the same trained ε_θ (DDIM shares
  DDPM's marginals q(x_t | x_0), so no retraining is needed).

## Diagnostic 1 — Terminal SNR of the assigned schedule

With T = 100 and β linear 1e-4 → 0.02, ᾱ_T = 0.3636 (signal coefficient
0.603), so x_T keeps ~60% of the image instead of being pure noise.

| x_T source | Mean | Std |
|---|---|---|
| Real q(x_T \| x_0) | -0.458 | 0.870 |
| N(0, I) used for sampling | +0.001 | 1.000 |

The network never sees N(0, I)-like inputs at t = 99, so sampling starts out of distribution.
Starting from real noised images instead (d_diag_real_xT.png), DDPM, DDIM-50 and DDIM-20 all produce
clean digits, which confirms the model and both samplers are correct. These are mostly reconstructions
of the source digits, not novel samples: DDPM differs from DDIM in only a few cells (stochastic noise
changing digit identity), while DDIM-50 and DDIM-20 match everywhere.

## Diagnostic 2 — Moment-matched x_T (no retraining, no real images)

x_T ~ N(√ᾱ_T · μ_pix, ᾱ_T · σ²_pix + 1 − ᾱ_T), using per-pixel MNIST statistics
(matched x_T: mean -0.445, std 0.879).

| Sampler | Time for 64 imgs (s) | Speed-up vs DDPM |
|---|---|---|
| DDPM    | 1.629 | 1.0x |
| DDIM-50 | 0.808  | 2.0x |
| DDIM-20 | 0.324  | 5.0x |

- DDPM output goes from scribbles to mostly recognizable digits (d_matched_xT_comparison.png).
- DDIM-50/20 also improve but stay thinner, more speckled and more often malformed. The per-pixel
  Gaussian ignores pixel correlations, so x_T is still slightly off-distribution; DDPM's per-step noise
  corrects for this, while deterministic DDIM carries the mismatch through to x_0.
- With no real image anchoring x_T, DDPM and DDIM reach different digits from the same start, while
  DDIM-50 and DDIM-20 still agree with each other.

## Ablation — Scaled schedule (β 1e-3 → 0.2, ᾱ_T ≈ 0), 20 epochs

Schedule = assigned β × 1000/T, i.e. the DDPM paper's T = 1000 schedule compressed to 100 steps.
Same x_T ~ N(0, I) and same seed as the main comparison.

| Sampler | Network evals | Time for 64 imgs (s) | Speed-up vs DDPM |
|---|---|---|---|
| DDPM    | 100 | 1.579 | 1.0x |
| DDIM-50 | 50  | 0.762  | 2.1x |
| DDIM-20 | 20  | 0.307  | 5.1x |

Final training loss 0.0222. This is not directly comparable to the main run's loss:
the schedules spend different shares of training at different noise levels.

- With ᾱ_T ≈ 0, N(0, I) is the correct start, and all three samplers produce clean, mostly
  recognizable digits (ablation_scaled_schedule_comparison.png).
- DDIM-50/20 reach near-DDPM quality with no speckle. The speckle under the assigned schedule came from
  the start mismatch, not from the DDIM sampler itself.
- DDIM-50 and DDIM-20 are nearly identical; DDIM-20 is slightly blobbier in a few cells. DDIM gives a
  2–5× speed-up at roughly equal quality.
- DDPM and DDIM reach different digits from the same x_T (stochastic vs. deterministic trajectories).
- A few malformed samples remain in all three, as expected for a small model trained 20 epochs.

## Summary

The assigned schedule leaves ~60% signal at t = T, so sampling from N(0, I) is out of distribution.
That explains why DDPM produced scribbles and DDIM produced speckle. Fixing the starting distribution
(moment-matched x_T) helps DDPM substantially but DDIM only partly. Fixing the schedule (ᾱ_T ≈ 0)
makes both work, with DDIM matching DDPM quality at 2–5× fewer network evaluations. That is the
intended DDPM → DDIM result.
