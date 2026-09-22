# AI_USE.md

## 1. Which parts I used AI for vs. wrote myself?

I used AI assistance to build the decoder only block and attention model. I had to perform the hyperparameter tuning myself to check for valid results. 

## 2. One specific wrong thing the AI produced?

In the initial run it suggested a different decoder block. Although the parameters were the same, I tried changing the dropout to get more coherent results. This did not help, so I tried identifying the root constraint in using the particular architecture. I observed that most of the issue with the results could be attributed to a small dataset. So I tried changing the decoder block, and did some hyperparameter tuning to achieve viable results.

## 3. How I found out it was wrong?

I checked the actual dataset and realized that it was a big constraint. I checked the decoder block and it was different from what was given in the model.

## 4. What I changed and why the fix works?

I changed the decoder block and tuned the parameters to get the current result.
