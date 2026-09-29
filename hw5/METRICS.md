
## Task 7: LoRA Rank Experiment

| Config | Trainable params | Final loss | ROUGE-1 | ROUGE-L |
|---|---|---|---|---|
| Baseline | 0 | - | 0.1456 | 0.1217 |
| r=4 (alpha=8) | 172,032 | 1.5002 | 0.3673 | 0.2921 |
| r=16 (alpha=32) | 688,128 | 1.4459 | 0.3719 | 0.2979 |

(Write your analysis here: how did output quality change between r=4 and r=16? Is the difference meaningful?)

## Task 6: Before vs. After Fine-Tuning

-Before fine-tuning, the model wrote in the first person and copied the dialogue's own wording ("I'd like to know if you can send over...", "I'm sorry, I don't want to start on the machines"). After fine-tuning, it produces third-person summaries in the reference style ("Mr. Blake tells Mr. Foster..."), so the format and tone moved closer to DialogSum. 
-The content is still weak, though: Dialogue 0 gets the details wrong (the manuals haven't been printed, and it says "tomorrow afternoon" instead of "this afternoon"). 
-Dialogue 1 shows repetition in both models, and the fine-tuned one confuses the speakers ("Johnson tells Johnson").
-Fine-tuning mainly fixed the style, not the understanding, which is expected for flan-t5-small with a short LoRA run.


LoRA fine-tuning improved ROUGE substantially over the baseline (ROUGE-1 0.146 -> 0.367 for r=4 and 0.372 for r=16; ROUGE-L 0.122 -> 0.292 and 0.298), 
so most of the gain comes from adapting the model at all, not from the choice of rank. 
Between ranks, r=16 (688,128 trainable params) was slightly better than r=4 (172,032) on every metric, 
but only by about 0.5 ROUGE-1 points and 0.6 ROUGE-L points, and it reached a slightly lower final training loss (1.446 vs 1.500). 
With only 50 evaluation examples, this gap is small enough that it may be within noise. 
Qualitatively, the ranks look similar: both produce third-person summaries but both loop in Dialogue 1, 
and r=4 added an unsupported detail in Dialogue 0 that r=16 did not. 
Overall, a 4x increase in rank gave at most a modest improvement, which suggests that for flan-t5-small on this short training run, 
adapter capacity is not the bottleneck.


## Task 7: Experiment Analysis

LoRA fine-tuning improved ROUGE substantially over the baseline (ROUGE-1 0.146 -> 0.367 for r=4 and 0.372 for r=16; ROUGE-L 0.122 -> 0.292 and 0.298), 
so most of the gain comes from adapting the model at all, not from the choice of rank. 
Between ranks, r=16 (688,128 trainable params) was slightly better than r=4 (172,032) on every metric, 
but only by about 0.5 ROUGE-1 points and 0.6 ROUGE-L points, and it reached a slightly lower final training loss (1.446 vs 1.500). 
With only 50 evaluation examples, this gap is small enough that it may be within noise. 
Qualitatively, the ranks look similar: both produce third-person summaries but both loop in Dialogue 1, 
and r=4 added an unsupported detail in Dialogue 0 that r=16 did not. 
Overall, a 4x increase in rank gave at most a modest improvement, which suggests that for flan-t5-small on this short training run, 
adapter capacity is not the bottleneck.
