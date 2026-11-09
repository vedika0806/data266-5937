# AI_USE.md

## 1. What I used AI for vs. wrote myself
- **AI :** Helped in the RAG pipelining code part: two chunking functions (fixed 150-word windows vs. sentence-aware), ChromaDB indexing and the conversational-memory class,
  Helped me infer the charts, and the evaluation code.
- **Me:** Performed data preprocessing, performed dense retrieval and cross-encoder reranking, set up the  workflow and parameters, reviewed the selected queries and retrieved chunks, compared the written findings against my actual result tables,
  and decided which conclusions the numbers actually support.

## 2. One specific wrong thing the AI produced
- I observed that:
- Fixed chunks improved: P@5 increased from 0.467 to 0.667.
- Sentence-aware chunks worsened: P@5 decreased from 0.667 to 0.600.
- Sentence-aware MRR also decreased from 0.833 to 0.733.
- MRR dropped from 1.0 to 0.2 under both strategies.
So there was a contention on whether the reranking really improved the retrieval. 
## 3. How I found out it was wrong
- Answered above

## 4. What I changed and why the fix works
- The code was not the problem, but there were reasons on why this happened which was likely:
-The relevance rubric used keywords.
-Sentence-aware chunks may have lost context.
-The corpus and queries were small and specialized.
-Reranking used only the top 30 dense results
