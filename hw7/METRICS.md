
## HW7: FashionMNIST Autoencoders

### Model Configuration

- Dataset: FashionMNIST
- FC-AE latent dimension: 64
- VAE latent dimension: 16
- Batch size: 128
- Epochs: 10
- Optimizer: Adam
- Learning rate: 0.001
- Device: cuda

### Results

| Model | Test MSE | Additional Metric |
|---|---:|---:|
| FC-AE | 0.011885 | Training time: 1.18 minutes |
| VAE | 0.014056 | Average KL loss: 16.241899 |

The FC-AE was evaluated using reconstruction mean squared error. The VAE was evaluated using deterministic reconstruction from the mean latent vector. The VAE also includes KL divergence to measure how closely its latent distribution follows the standard normal distribution.

