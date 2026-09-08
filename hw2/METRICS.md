
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


## Q3: Optimization Techniques Exploration

### Shared Experimental Setup
Same model (4-layer MLP, hidden_dim=1024), same synthetic data (seed=5937),
batch_size=64, 50 training steps.

An initial run showed peak GPU memory increasing monotonically across every experiment regardless of technique. Some techniques (e.g.checkpointing) should reduce memory instead of increase. Since the previous experiments were not removed from the GPU memory, so each successive experiment's measurement
sat on top of leftover allocations rather than a clean state.
Adding `del model` + `torch.cuda.empty_cache()` between each CUDA experiment helped with the memory issue and it started decreasing as expected.More isolated experiments can get better results which would eliminate the run-order influence of the process.

### Results Table
| technique           |   elapsed_sec |   peak_gpu_mem_MB |   final_loss |   avg_loss |   tensor_creation_cpu_sec |   tensor_creation_gpu_sec |
|:--------------------|--------------:|------------------:|-------------:|-----------:|--------------------------:|--------------------------:|
| default_init        |        0.08   |            215.66 |       2.3041 |     2.3023 |                   40.8204 |                    0.7737 |
| xavier_init         |        0.0839 |            205.12 |       2.315  |     2.3383 |                   40.8204 |                    0.7737 |
| no_checkpointing    |        0.08   |            194.58 |       2.3041 |     2.3023 |                   40.8204 |                    0.7737 |
| with_checkpointing  |        0.1998 |            184.04 |       2.3041 |     2.3023 |                   40.8204 |                    0.7737 |
| no_grad_accum       |        0.0759 |            173.5  |       2.3041 |     2.3023 |                   40.8204 |                    0.7737 |
| grad_accum_4steps   |        0.0687 |            156.49 |       2.3048 |     2.3023 |                   40.8204 |                    0.7737 |
| fp32                |        0.075  |            141.88 |       2.3041 |     2.3023 |                   40.8204 |                    0.7737 |
| mixed_precision_amp |        0.1152 |            131.21 |       2.3042 |     2.3023 |                   40.8204 |                    0.7737 |

### Findings
**Tensor Creation (CPU vs GPU):** CPU took 40s where as GPU took 0.77s for 50iterations of 5000x5000 matrix creation + matmul — roughly a 53x speedup.This is the expected result since GPUs are built for larger applications done parallely in contrast to a CPU which runs a single thread.
**Weight Initialization:** Default init reached final loss 2.3041 vs Xavier init at 2.3150 — essentially identical (and Xavier's avg_loss, 2.3383, was actually slightly
higher).Since the setup is such that the data has random noise and no real signals to learn from, such result was expected. In real data, Xavier will give smoother early gradients which would benefit in deeper networks.
**Activation Checkpointing:** Time went from 0.0800s (no checkpointing) to 0.1998s (with
checkpointing) — about 2.5x slower, matching the time-memory tradeoff of recomputing forward activations during backward instead of storing them. Peak memory
went from 194.58MB to 184.04MB, which is modest given the model, however, can give better results in deeper networks. 
**Gradient Accumulation:** No-accumulation final loss 2.3041 vs 4-step accumulation final
loss 2.3048 since real data exposure is less. Peak memory also decreased slightly (173.50MB -> 156.49MB) after cleanup, consistent with accumulation processing smaller per-step batches under the hood.
**Mixed Precision Training:** FP32 took 0.0750s vs AMP 0.1152s — AMP was slower here.
Peak memory went from 141.88MB (fp32) to 131.21MB (AMP), a decrease in the expected
direction.
