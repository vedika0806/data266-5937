
## Task 7: LoRA Rank Experiment

| Config | Trainable params | Final loss | ROUGE-1 | ROUGE-L |
|---|---|---|---|---|
| Baseline | 0 | - | 0.1456 | 0.1217 |
| r=4 (alpha=8) | 172,032 | 1.5002 | 0.3673 | 0.2921 |
| r=16 (alpha=32) | 688,128 | 1.4459 | 0.3719 | 0.2979 |

(Write your analysis here: how did output quality change between r=4 and r=16? Is the difference meaningful?)
