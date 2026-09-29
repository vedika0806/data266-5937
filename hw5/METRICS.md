
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
