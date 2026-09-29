# AI_USE.md
1. What I used AI for vs. wrote myself?
- I used AI to understand LORA finetuning and the coding parameters and the impact analysis. I worked on the ROGUE finetuning myself. 
2. One specific wrong thing the AI produced (paste actual bad output/error)?
-I read the printed outputs for Dialogue 1 and saw the same sentence repeated several times for r=4, r=16, and the baseline, even though ROUGE had improved. 
Since the baseline looped too, I concluded the cause was decoding not LoRA.
3. How I found out it was wrong?

- I read the printed outputs for Dialogue 1 and saw the same sentence repeated several times
4. What I changed and why the fix works?
- I did not change the decoding settings. I identified that adding no_repeat_ngram_size=3 to model.generate() would likely reduce the looping, but I chose to keep the original settings so the ROUGE numbers and sample outputs in METRICS.md all come from the same setup and stay consistent. 
