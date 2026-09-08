
## Q1: Embedding-Based Transfer Learning (Word2Vec on IMDB)

### Top-3 Nearest Neighbors — Before Fine-Tuning
| word   |   rank | neighbor   |   cosine_sim | stage   |
|:-------|-------:|:-----------|-------------:|:--------|
| cast   |      1 | casts      |       0.7219 | before  |
| cast   |      2 | casting    |       0.7188 | before  |
| cast   |      3 | Cast       |       0.6638 | before  |
| score  |      1 | scoring    |       0.7197 | before  |
| score  |      2 | scores     |       0.6596 | before  |
| score  |      3 | scored     |       0.6384 | before  |
| plot   |      1 | plots      |       0.7625 | before  |
| plot   |      2 | Plot       |       0.6524 | before  |
| plot   |      3 | plotting   |       0.6328 | before  |
| screen |      1 | screens    |       0.7729 | before  |
| screen |      2 | onscreen   |       0.6115 | before  |
| screen |      3 | LCD_screen |       0.5599 | before  |
| review |      1 | reviewed   |       0.663  | before  |
| review |      2 | reviewing  |       0.661  | before  |
| review |      3 | reviews    |       0.638  | before  |

### Top-3 Nearest Neighbors — After Fine-Tuning
| word   |   rank | neighbor       |   cosine_sim | stage   |
|:-------|-------:|:---------------|-------------:|:--------|
| cast   |      1 | actors         |       0.5881 | after   |
| cast   |      2 | performances   |       0.5166 | after   |
| cast   |      3 | casting        |       0.5026 | after   |
| score  |      1 | soundtrack     |       0.6329 | after   |
| score  |      2 | composer       |       0.566  | after   |
| score  |      3 | cinematography |       0.5585 | after   |
| plot   |      1 | storyline      |       0.656  | after   |
| plot   |      2 | story          |       0.5999 | after   |
| plot   |      3 | plots          |       0.5905 | after   |
| screen |      1 | screens        |       0.5396 | after   |
| screen |      2 | stage          |       0.4533 | after   |
| screen |      3 | lebowski       |       0.414  | after   |
| review |      1 | comment        |       0.6896 | after   |
| review |      2 | reviews        |       0.5777 | after   |
| review |      3 | imdb           |       0.5619 | after   |

### Per-Word Cosine Similarity: Original vs Fine-Tuned Vector
| word   |   cosine_sim_before_vs_after |
|:-------|-----------------------------:|
| review |                       0.5268 |
| score  |                       0.5944 |
| screen |                       0.6745 |
| plot   |                       0.7274 |
| cast   |                       0.7604 |

**Most shifted word:** review
**Least shifted word:** cast

t-SNE visualization saved as `tsne_plot_shift.png`.

### Analysis
Before fine tuning, the nearest neighbours are just other forms of the same word- like "cast" is similiar to "casts", "casting" and so on. Since Google News is not trained on movie reviews, it is grouping the spelling/grammar words together.

After fine tuning, the nearest neighbours are more semantically similar, like cast is closer to actors and performances, instead of casting. "Score" is closer to cinematography, composer etc. "Plot" is closer to storyline, review etc.

"Review" shifted the most out of all five words because it might mean different things in general news but is very unambiguous in IMdb dataset.
"Cast" did not move much, probably because it had the same sense in News as well.

## Q2: RAG Pipeline 

### Questions, Retrieved Chunks, and Answers

**Q1: Who directed Inception?**
Top chunk directly named Christopher Nolan as writer/director. Answer: "Christopher Nolan." Correct, rank 1.

**Q2: What crime family is central to The Godfather?**
Top chunk named the Corleone family directly. Answer: "The Corleone mafia family." Correct, rank 1.

**Q3: What real-world issue does Parasite explore?**
Answer: "I don't know." The LLM correctly avoided hallucinating given weak context, but retrieval failed to surface the relevant content.

**Q4: Who plays the Joker in The Dark Knight?**
Rank 1 chunk listed the full cast without pairing actors to roles; rank 2 chunk referenced
"Ledger's casting... to portray the Joker," which is what actually answered the question.
Answer: "Heath Ledger." Correct, but the truly explicit answer was rank 2, not rank 1.

**Q5: What nonlinear storytelling technique does Pulp Fiction use?**
Answer described the 7-part spliced structure correctly. Correct, rank 1.

### Chunk Size Comparison (500/50 vs 250/25)
Re-ran Q1 and Q2 with a smaller chunk size (250 chars, 25 overlap). Final answers were
identical in both cases ("Christopher Nolan"; "The Corleone family"), so chunk size didn't
change the outcome here. What did change was chunk composition: under the smaller config,
The Godfather's rank-1 chunk trimmed down to just the cast list and trilogy mention, while
the phrase that actually names "the Corleone family" under patriarch Vito Corleone moved to
chunk 2 instead of chunk 1. This shows that smaller chunks can separate a name from the
sentence that gives it context, splitting information that was previously together in one
chunk. It didn't hurt the final answer here only because the LLM still saw both chunks
together in its context window — with a smaller top-k, this could have caused a real failure.

### Retrieval Evaluation
| question                                                     | correct_answer_found_in_top3   |   rank_of_first_relevant_chunk |
|:-------------------------------------------------------------|:-------------------------------|-------------------------------:|
| Who directed Inception?                                      | Yes                            |                              1 |
| What crime family is central to The Godfather?               | Yes                            |                              1 |
| What real-world issue does Parasite explore?                 | No                             |                            nan |
| Who plays the Joker in The Dark Knight?                      | Yes                            |                              2 |
| What nonlinear storytelling technique does Pulp Fiction use? | Yes                            |                              1 |

**Retrieval Success Rate: 0.8** (4 out of 5 questions had a relevant chunk in the top 3)


## RAG Failure Analysis
**Failure 1: Correct document retrieved, but relevant chunk never surfaced (Parasite)**
Question: "What real-world issue does Parasite explore?"
The chunks were retrieved from the correct document, but none of them were pointing to the film's thematic content, so the LLM denied to answer.However, it was still a retrieval failure instead of LLM failure because actual thematic
content likely exists elsewhere on the page that didn't rank in the top 3 for this query's embedding.

**Failure 2: Correct answer present but not in the top-ranked chunk (The Dark Knight)**
Question: "Who plays the Joker in The Dark Knight?"
Chunk 1 lists the full cast but never pairs any actor to a specific role. Chunk 2 is the chunk that actually implies the answer. The LLM was still able to answer correctly since both chunks were in its context window, but the truly explicit answer wasn't the top-ranked chunk.

