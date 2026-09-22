
## Model & Training Configuration (final submitted run)

| Item | Value |
|---|---|
| Dataset | Shakespeare.txt (Romeo & Juliet balcony scene excerpt, ~2,050 characters) |
| Vocabulary size | 49 |
| Sequence length | 128 (per assignment spec) |
| Hidden dimension (d_model) | 128 |
| Number of heads | 4 |
| Number of layers | 2 |
| Total parameters | 425,777 |
| Optimizer | Adam |
| Learning rate | 4e-4 |
| Batch size | 32 |
| Epochs | 8 |
| Attention design | Pre-norm (x = x + attn(LayerNorm(x)))|

## Training Loss per Epoch (SEQ_LEN=128 final run)

| Epoch | Loss |
|---|---|
| 1 | 2.9987 |
| 2 | 2.3795 |
| 3 | 2.2104 |
| 4 | 2.1259 |
| 5 | 2.0496 |
| 6 | 1.9569 |
| 7 | 1.8471 |
| 8 | 1.7114 |

Curve shape: smooth, monotonic decrease across all 8 epochs, no spikes or plateau. See `training_loss_curve.png`.

## Permutations Tried During Development

Several configurations were tried during development of the miniGPT before settling on the final run given above:
**1. Initial run — SEQ_LEN=128, STRIDE=8, D_MODEL=256, N_HEADS=4, N_LAYERS=4, post-norm blocks, LR=3e-4, BATCH=64, 6 epochs.**
Result: Extremely repetitive greedy output and very low diversity. It was an oversized model given the tiny dataset combined with few gradient steps.

**2. Reduced model size, changed windowing — SEQ_LEN=64, STRIDE=2, D_MODEL=128, N_HEADS=4, N_LAYERS=2, LR=6e-4, BATCH=64, 8 epochs.**
Result: Real words and structure began appearing, understood that the tiny dataset was the root constraint. We cannot hold the result because the Sequential length is required to be 128 as per the assignment.and

**3. Reverted SEQ_LEN to 128 and corrected DecoderBlock from post-norm to pre-norm and added a final LayerNorm before the output head**
Result: Final loss is 1.711 higher than what was achieved for SEQ_LEN = 64 because of the constraints of the dataset and hence the output is slightly more repetitive/loopy.



## Decoding Method Comparison

| Method | Distinct-word ratio | Real-word ratio |
|---|---|---|
| Greedy | 0.638 | 0.207 |
| Temperature 0.3 | 0.796 | 0.204 |
| Temperature 0.8 | 0.899 | 0.261 |
| Temperature 1.5 | 0.981 | 0.093 |
| Top-k (k=3) | 0.863 | 0.196 |
| Top-k (k=40) | 0.967 | 0.148 |



Diversity increases monotonically as expected: greedy < T=0.3 < T=0.8 < T=1.5, and k=3 < k=40. Real-word ratio is not perfectly monotonic (T=0.8 exceeds T=0.3)

## Decoding Strategy Analysis

**Greedy decoding** always selects the argmax character. It is very deterministic and has the lowest diversity. It has memorised some fragments and is looping on the same instead of generating new content.

**Temperature sampling** : T = 0.3 stays closer to the source text with heavy repetition (diversity 0.796). T=0.8 produces more variety (0.899) and the highest real world ratio likely because of moderate randomness breaks reptition loops. T = 1.5 gives very diverse output and lowest real world ratio. 

**Top-k sampling** restricts sampling to k most probable characters. k=3 stays closer to confident predictions with more variety than greedy (0.863). k = 40 allows a wider pool increasing diversity at the cost of coherence.

**Most coherent output:** T = 0.8 is the most coherent output as it has the highest real-word ratio (0.261). 

**Most diverse output:** Temperature = 1.5 has the highest distinct-word ratio (0.981), but is nearly unreadable noise


## Architecture Note

The decoder block uses a pre-norm design (`x = x + attn(LayerNorm(x))`, then `x = x + ffn(LayerNorm(x))`), matching the course demo notebook's design, extended to multi-head causal self-attention (4 heads, manual implementation — no `nn.MultiheadAttention` or `nn.Transformer` used). A final LayerNorm is applied before the output projection.
