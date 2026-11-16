
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


## Findings

### Part 1 findings: caption "an image of a horse" vs. 100 CIFAR-10 test images per class

- **The caption picks out its class.** Horse has the highest mean cosine similarity (0.2757 ± 0.0155), ahead of deer (0.2496).
  The gap of 0.026 is about 1.7 standard deviations of the horse scores.
  - Precision@100 = 81% (chance = 10%): of the 100 images most similar to the caption, 81 are horses.
  - ROC-AUC = 0.972: a random horse image outscores a random non-horse image about 97% of the time.
- **The ranking follows meaning, not just pixels.** Deer is clearly second, and it supplies 15 of the 19 non-horse images in the top 100
  (the rest: bird 3, cat 1). Deer are the most horse-like class: four-legged, brown, grass and forest backgrounds.
  The other animals (frog, dog, cat, bird) form a middle band (0.229–0.231). All four vehicle classes come last (airplane 0.216, truck 0.208,
  automobile 0.204, ship 0.202). So CLIP's embedding space puts animal images closer to "horse" than vehicle images.
- **Absolute scores are small and tightly packed.** All 1,000 cosine values fall between about 0.16 and 0.30, and the class means differ by at most
  about 0.07. CLIP image–text cosines are usually in this range, so the ranking carries the information, not the raw value.
  The overlap is real: the lowest-scoring horse (0.202) scores below the average airplane, likely a blurry or unusual 32×32 image.
- **Why no softmax:** with a single caption, softmax always returns 1.0, so raw cosine similarity is the only meaningful score here.
  CLIP's learned temperature (logit_scale) only rescales it.

### Part 2 findings: zero-shot classification on all 10,000 CIFAR-10 test images (prompt "an image of a {class}")

- **Overall accuracy is 89.06%** (macro F1 = 0.890) against 10% chance, with no training on CIFAR-10.
  The class names alone, turned into text prompts, act as the classifier.
- **Vehicles are easier than animals.** Truck 96.8%, ship 95.7%, automobile 93.7% and airplane 89.5% average about 94%.
  The six animal classes average about 86%. Horse (96.8%) is the exception among animals.
  The hardest classes are frog (75.7%) and deer (78.9%).
- **The biggest confusions are pairs that look alike or are semantically related:**
  deer→horse (132), frog→cat (95), cat→dog (81), dog→cat (73), frog→bird (66), automobile→truck (56), airplane→bird (47).
  Cat↔dog runs in both directions (81 / 73). Automobile→truck (56) is twice as common as truck→automobile (26).
  Airplane→bird (47) makes sense: a small object against a plain sky looks similar at 32×32.
- **Precision and recall show which classes absorb errors:**
  - Frog has the highest precision (0.979) but the lowest recall (0.757). CLIP rarely says "frog" unless it is sure,
    and its missed frogs spread out to cat, bird and dog.
  - Horse is the opposite (recall 0.968, precision 0.832). It absorbs other animals, especially 132 deer.
  - Cat also absorbs errors (precision 0.796), from frogs and dogs.
- **Confidence is not reliability.** The most confident wrong predictions have softmax probabilities of 0.97–0.99:
  dog→deer, ship→airplane (twice), deer→horse (twice), bird→deer.
  CLIP's large logit scale sharpens the softmax, so a wrong answer can still look almost certain.
- **Low resolution plays a role.** CIFAR images are 32×32 and get upsampled to 224×224 for ViT-B/32. Fine details such as antlers
  or the face shape that separates a cat from a dog are mostly lost, which fits with the worst confusions being between small animals.

### Connecting the two parts
- Part 1's results predict Part 2's. The class closest to "an image of a horse" in Part 1 (deer) is the source of the largest
  confusion in all of Part 2 (132 deer predicted as horse).
- Part 1's high Precision@100 (81%) and AUC (0.97) for horse match horse's high zero-shot recall (96.8%).
- The 15 deer among the top 100 horse-like images in Part 1 explain why horse's precision (0.832) is much lower than its recall.
- The same embedding geometry explains both experiments: deer images sit near the "horse" text embedding.

## Part 2 — Zero-shot classification (CIFAR-10)

Model: `openai/clip-vit-base-patch32` | Dataset: CIFAR-10 test set, all 10,000 images | No training (zero-shot)
Prompts: `an image of a {}` for each class | Prediction = argmax cosine similarity

**Overall accuracy: 89.06%**

### Per-class accuracy
| class      |   correct |   total | accuracy   | most_confused_with   |
|:-----------|----------:|--------:|:-----------|:---------------------|
| airplane   |       895 |    1000 | 89.50%     | bird                 |
| automobile |       937 |    1000 | 93.70%     | truck                |
| bird       |       875 |    1000 | 87.50%     | horse                |
| cat        |       870 |    1000 | 87.00%     | dog                  |
| deer       |       789 |    1000 | 78.90%     | horse                |
| dog        |       890 |    1000 | 89.00%     | cat                  |
| frog       |       757 |    1000 | 75.70%     | cat                  |
| horse      |       968 |    1000 | 96.80%     | dog                  |
| ship       |       957 |    1000 | 95.70%     | airplane             |
| truck      |       968 |    1000 | 96.80%     | automobile           |

### Confusion matrix (rows = true, columns = predicted)
|                  |   airplane |   automobile |   bird |   cat |   deer |   dog |   frog |   horse |   ship |   truck |
|:-----------------|-----------:|-------------:|-------:|------:|-------:|------:|-------:|--------:|-------:|--------:|
| true: airplane   |        895 |            6 |     47 |     0 |      0 |     3 |      0 |       1 |     29 |      19 |
| true: automobile |          3 |          937 |      0 |     0 |      0 |     1 |      0 |       0 |      3 |      56 |
| true: bird       |         10 |            1 |    875 |    28 |     28 |    18 |      5 |      32 |      3 |       0 |
| true: cat        |          3 |            4 |     17 |   870 |      8 |    81 |      5 |       8 |      3 |       1 |
| true: deer       |          3 |            1 |     22 |    20 |    789 |    26 |      5 |     132 |      2 |       0 |
| true: dog        |          0 |            1 |     12 |    73 |      9 |   890 |      1 |      12 |      2 |       0 |
| true: frog       |          1 |            3 |     66 |    95 |     19 |    39 |    757 |      10 |     10 |       0 |
| true: horse      |          4 |            0 |      4 |     6 |      7 |    10 |      0 |     968 |      0 |       1 |
| true: ship       |         26 |            5 |      2 |     1 |      0 |     1 |      0 |       0 |    957 |       8 |
| true: truck      |          0 |           26 |      0 |     0 |      0 |     0 |      0 |       0 |      6 |     968 |

### Top confusions
| true       | predicted   |   count |
|:-----------|:------------|--------:|
| deer       | horse       |     132 |
| frog       | cat         |      95 |
| cat        | dog         |      81 |
| dog        | cat         |      73 |
| frog       | bird        |      66 |
| automobile | truck       |      56 |
| airplane   | bird        |      47 |
| frog       | dog         |      39 |


