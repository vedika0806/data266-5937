
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
