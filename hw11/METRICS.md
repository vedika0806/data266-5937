
## Part 1 — Caption vs. images similarity (CIFAR-10)

Model: `openai/clip-vit-base-patch32` | Dataset: CIFAR-10 test set, 100 images/class × 10 = 1000 images (seed=5937)
Caption (CLS_A=7): **"an image of a horse"** | Score = cosine similarity of L2-normalized CLIP embeddings

### Per-class similarity to the caption
| class      |   mean |    std |    min |    max |   rank |
|:-----------|-------:|-------:|-------:|-------:|-------:|
| horse      | 0.2757 | 0.0155 | 0.2019 | 0.3003 |      1 |
| deer       | 0.2496 | 0.0157 | 0.1979 | 0.2854 |      2 |
| frog       | 0.2308 | 0.013  | 0.1928 | 0.2596 |      3 |
| dog        | 0.2304 | 0.0129 | 0.2066 | 0.2645 |      4 |
| cat        | 0.2294 | 0.0121 | 0.194  | 0.2722 |      5 |
| bird       | 0.2286 | 0.0154 | 0.2008 | 0.2742 |      6 |
| airplane   | 0.2157 | 0.0132 | 0.1861 | 0.2495 |      7 |
| truck      | 0.2083 | 0.0093 | 0.1835 | 0.2353 |      8 |
| automobile | 0.2038 | 0.0114 | 0.1766 | 0.2369 |      9 |
| ship       | 0.2018 | 0.0132 | 0.1629 | 0.229  |     10 |

### Does the caption single out its class?
| Metric | Value | Chance |
|---|---|---|
| Target class rank by mean similarity | 1 / 10 | — |
| Precision@100 | 81.00% | 10% |
| ROC-AUC (target vs. rest) | 0.9723 | 0.5 |
| Most similar non-target class | deer | — |

Class mix of the 100 most similar images: horse: 81, deer: 15, bird: 3, cat: 1

