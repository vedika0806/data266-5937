# AI_USE.md

## Which parts I used AI for and which I wrote myself
- I used AI to understand the U-Net design and the classifier-free guidance sampler and to understand the results I got. 
## One specific wrong thing the AI produced (paste the actual output)
- Initially, it suggested no optimizer while training, which would have impacted the loss or made the training longer. I added the Adam optimizer for effective training and loss assessment.
## How I found out it was wrong
- I could not find any optimizer in the training suggestions whilst designing the architecture.
