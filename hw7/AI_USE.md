
# AI_USE.md

## How AI was used?

AI assistance was used to help understand the structure of the autoencoder & VAE, and troubleshoot code during development. The final implementation was run and tested in Google Colab.

## Any specific problem encountered?

AI initially described these results as good based mainly on the numerical metrics. However, after I inspected the reconstructed images, I noticed that they were blurry and did not reproduce the original clothing images very well.

## How I identified the problem

The numerical MSE value alone did not fully represent the visual quality of the reconstructions. By comparing the original FashionMNIST images with the reconstructed images, I noticed that important details and shapes were missing.

## What I changed
I learned that increasing the number of training epochs, increasing the VAE latent dimension, and reducing the weight of the KL-divergence term can help give more solid and good reconstructed images. 

