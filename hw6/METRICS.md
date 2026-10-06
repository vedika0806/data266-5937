
## STL-10, ResNet-18 from scratch, 500 labels (SID4=5937, SEED=5937)

Personal parameters: SID4=5937, SEED=5937, SLICE=937, HP_ID=3, CLS_A=7, CLS_B=6
(only SEED is used in this homework; the HP_ID second-model requirement is waived).

| Model | Pretraining data / epochs | Eval protocol | Test accuracy | Top-5 NN precision (cosine) |
|---|---|---|---|---|
| A. Supervised | 500 labels, 15 epochs end-to-end | end-to-end | 0.4529 | 0.3894 |
| B. Rotation SSL | 100k unlabeled, 15 epochs (pretext rot-acc 0.871) | frozen encoder + linear, 20 epochs | 0.5266 | 0.4476 |
| C. SimCLR (tau=0.2) | 20k unlabeled, 20 epochs, batch 256 | frozen encoder + linear, 20 epochs | 0.5319 | 0.4761 |

Chance = 0.10. Reported accuracies are final-epoch (no best-epoch selection on the test set).
Reproduce: run the notebook top to bottom on a Colab GPU (seeded; cuDNN deterministic mode on, AMP fp16 so tiny run-to-run differences are possible).
Checkpoints: `checkpoints/{supervised,rotation,simclr}_encoder.pt`. Query test indices for Part D: [58, 4255, 2414, 4274].

## Discussion (SID4=5937, SEED=5937; single run, one seed)

| Model | Test acc | Top-5 NN precision | Chance |
|---|---|---|---|
| A. Supervised (500 labels) | 0.4529 | 0.3894 | 0.10 |
| B. Rotation SSL + linear | 0.5266 | 0.4476 | 0.10 |
| C. SimCLR + linear | 0.5319 | 0.4761 | 0.10 |

### 1. Which model performed best, and why?
SimCLR was best on both metrics, but the two self-supervised models are close on accuracy. C beats B by only 0.53 points
(0.5319 vs 0.5266), which is smaller than the epoch-to-epoch swing in the probes' test accuracy (B ranged about 0.496-0.539 and
C about 0.519-0.538 over epochs 4-20), and this is one seed. I treat B and C as roughly tied on accuracy. SimCLR's lead is clearer
on top-5 NN precision (0.4761 vs 0.4476, +2.9 points), which does not depend on a trained probe, though it is still one seed.
Both SSL models beat the supervised baseline by about 7.4 (B) and 7.9 (C) points of accuracy, and by 5.8 and 8.7 points of NN precision.
The main reason is data: the SSL encoders learned from 100k (B) and 20k (C) unlabeled images, while A saw only 500 labelled ones.
A plausible reason SimCLR does well on NN retrieval is that NT-Xent trains embeddings under cosine similarity, which is the
metric used for retrieval. I did not test this directly.

### 2. Why did the other models perform as they did?
- Supervised (A): training loss fell from 2.34 to 0.50 over 15 epochs, but test accuracy flattened at about 0.43-0.45 from epoch 11
  on (0.4373 -> 0.4529). A falling loss with a flat test score points to memorising 500 images with an ~11M-parameter network.
  I did not log train accuracy, so the gap is inferred from the loss. A also has the lowest NN precision (0.3894, below its
  own accuracy of 0.4529), suggesting its embedding space was shaped for the final linear head rather than for general similarity.
- Rotation SSL (B): the pretext task was learned well (final rotation accuracy 0.871 vs 0.25 chance), but a good pretext score
  does not guarantee class-relevant features. Rotation can be solved from orientation cues (sky/ground, horizon, gravity-aligned
  structure), which only partly separate STL-10 classes. B reaches 0.5266 and is the weakest of the two SSL models on NN precision.
- SimCLR (C): it used only 20k images, batch 256 and 20 epochs, which is about 1,560 gradient steps (78 per epoch) versus 5,850
  for rotation (390 per epoch x 15). It still matched or beat rotation, which suggests the contrastive objective extracts more
  per update (a hypothesis). SimCLR normally benefits from far longer training and larger batches than this.
- Both linear probes overfit their 500 labels: final train accuracy 0.944 (B) and 0.966 (C) against test accuracy of about 0.53.
  So the 500-label probe, not only the representation, limits the SSL numbers.

### 3. What changes could improve each model?
- A: stronger augmentation (mixup/cutmix, random erasing, colour jitter), label smoothing, tuned weight decay, and a validation
  split carved from the remaining training images (not the test set) for early stopping.
- B: more pretext epochs, a harder pretext task, and probing intermediate layers (e.g. layer3) because the last block
  specialises in rotation. Stronger augmentation during the pretext could also help.
- C: use all 100k unlabeled images, train for 100+ epochs, use a larger batch (512-1024), stronger crop/colour augmentation,
  tune tau (0.2 vs 0.1/0.5), and fine-tune instead of freezing.
- All probes: stronger regularisation or early stopping (selected on a validation split), and several seeds, since the B-vs-C gap
  is within noise.

### 4. What did the nearest-neighbour figures reveal about representation quality?
Quantitatively, over all 8,000 test queries, top-5 precision was 0.3894 (A), 0.4476 (B) and 0.4761 (C), against 0.10 for random.
Every encoder retrieves the wrong class for more than half of its neighbours, so none of the three representations is strongly
class-aligned, which fits their modest accuracies. The ranking A < B < C matches the accuracy ranking.
Qualitatively, query indices [58, 4255, 2414, 4274] (shared by all encoders):
Query 1 (idx 58, car): green neighbours A= 4/5, B= 3/5, C= 3/5
Query 2 (idx 4255, airplane): A= 1/5, B= 4/5, C= 3/5
Query 3 (idx 2414, truck): A= 1/5, B= 2/5, C= 0/5
Query 4 (idx 4274, horse): A= 1/5, B= 3/5, C= 4/5
Totals across the 4 queries: A= 7/20, B= 12/20, C= 10/20
On these four queries the three encoders do not follow the full-test-set ranking: the totals are A=7/20, B=12/20 and C=10/20,
so B beats C here even though C is higher on 8,000-query precision (0.4761 vs 0.4476). No encoder wins on every query:
A is best on the car query (4/5), B on the airplane query (4/5), and C on the horse query (4/5), while C retrieves no
same-class neighbours for the truck query (0/5). With only four queries, a single query changes the totals by several points,
so I treat this as an illustration of variability across queries, not a ranking, and rely on the 8,000-query precision
(A < B < C) as the evidence. A is lowest in both views, consistent with it having the weakest embedding space.
