
## HW9 Results: Latent Diffusion on Fashion-MNIST

| Item | Value |
|---|---|
| VAE | stabilityai/sd-vae-ft-mse (pretrained, frozen) |
| Image size / latent shape | 256x256 / 4x32x32 |
| Latent scaling factor | 0.18215 |
| Scaled latent mean / std (train) | -0.0245 / 1.3606 |
| Reconstruction MSE, 1,000 test images, [-1, 1] | 0.000125 |
| Reconstruction MSE, 1,000 test images, [0, 1] | 0.000031 |
| Diffusion | T=1000, linear beta 0.0001 to 0.02 |
| U-Net | base 64, mult (1, 2, 4), attention at 16x16 + bottleneck, 14.74 M params |
| Training | 50 epochs, Adam, LR 2e-05, batch 64, label dropout 0.1 |
| Final training loss | 0.05001 |
| Sampling | DDPM 1000 steps, guidance w=3, 64 images |

### Analysis

**VAE reconstruction.** On 1,000 test images, the reconstruction MSE was 0.000125 on the [-1, 1] scale, or
0.000031 on the [0, 1] scale. The reconstructions in p1_vae_reconstruction.png are almost indistinguishable from
the originals: garment shapes and fine details such as bag handles, shoe soles and hood outlines are preserved.
This shows that the 4x32x32 latent keeps enough information for diffusion to work in latent space, even though
the VAE was pretrained on natural images rather than Fashion-MNIST. It also means the softness of the generated
samples does not come mainly from VAE compression.

**Forward diffusion.** p2_forward_diffusion.png uses one image and the same noise at every timestep. At t = 0
(abar = 1.000) the decoded image looks the same as the clean one. At t = 250 (abar = 0.521) the boot's silhouette
is still recognizable, but its surface detail is lost and the background is corrupted. By t = 500 (abar = 0.078)
only a faint trace of the shape remains, and at t = 750 and t = 999 no trace is visible. The noise is added to the
latents rather than the pixels, so the decoded images show mottled, blob-like VAE artifacts rather than plain
per-pixel static. At t = 999, abar rounds to 0.000, so z_t is almost pure Gaussian noise. That is the starting
point the reverse process needs.

**Training.** The model trained for all 50 epochs (about 46,850 updates) at about 158 s per epoch, or roughly
2.2 hours on the GPU. Loss fell from 0.0572 at epoch 20 to 0.0500 at epoch 50. It was still falling slowly at
the end, with small fluctuations between epochs (0.0497 at epoch 49, 0.0500 at epoch 50). Training was not
fully converged at this low learning rate of 2e-5, so more epochs might bring a small improvement. The runtime
disconnected after training finished, so the final model was reloaded from the saved checkpoint for sampling.

**Conditional samples.** With guidance w = 3, all 64 samples in the grid are recognizable as their requested
class, which shows that label conditioning and classifier-free guidance are working. T-shirts, trousers, boots
and sneakers are the most consistent, with clear silhouettes and some variety, such as printed T-shirts and
both flat and heeled boots.

**Weaker classes.** Sandals are the weakest class: their thin straps give fragmented or blobby shapes in several
samples, probably because small structures are hard to preserve in a latent space that is downsampled 8 times.
Bags and some pullovers are often nearly solid white with little interior detail, and 3 of the 6 bags are
close to plain rectangles. This may be a saturation effect from the strong guidance, which pushes samples
toward high contrast. Shirts, pullovers and coats look similar to each other, which is expected because these
classes have similar shapes in Fashion-MNIST.

**Image sharpness.** Edges look soft. The source images are only 28x28 and were enlarged to 256x256 for the VAE,
so the higher resolution adds no real detail.

**Limitation.** This assessment is visual and based on 64 samples. No quantitative metric such as FID was computed.

