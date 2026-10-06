# AI_USE.md

## 1. What I used an assistant for vs. wrote myself
- I used AI assistance for the rotation self supervised learning part because that was new to me. I also worked with AI to understand the interpretation of the results I got from Part a, b & c. 

## 2. One specific wrong thing the assistant produced?
- It misinterpreted the representation quality with the query.

## 3. How I found out (what did the failure look like?)
- I reevaluated the interpretations to identify which encoders performed consistently and for which the results were departed from expectations. It suggested that C misperformed, but after studying the overall results, I observed that the 8,000-query number is the one to trust, and the 4-query result is just a few examples. 
So, C has not departed from its expectations.
- 
