# AI_USE.md
## 1. Which parts did you use an assistant for, and which did you write yourself?

-I used AI assistant for scaffolding the cosine similarity part for Q1. 
-For the FAISS part in Q2 and usage of Ollama over Colab.
-To solve the memory cache issue for Q3


## 2. Give one specific thing it produced that was wrong — a tensor shape error, a
deprecated API, a loss function that trained but was wrong, a plausible-looking
metric computed incorrectly. Paste the wrong output.

While searching for the relevant datasets, it gave the following error: HfUriError: Invalid HF URI 'hf://datasets/imdb@e6281661ce1c48d982bc483cf8a173c1bbeb5d31/.huggingface.yaml'.
Repository id must be 'namespace/name', got 'imdb'. I switched to load_dataset('stanfordnlp/imdb') after HF Hub renamed the dataset repo.
While running the optimization experiments for Q3, I noticed that the memory usage was monotonically increasing, which was a departure from the conventional experiment beahviour and outcomes. 
I switched on to the cache memory part on the advise of AI. 

## 3. How did you find out? What did the failure look like?
peak GPU memory increased for every single experiment regardless of technique (131.34MB -> 216.44MB across 8 experiments run in sequence)
## 4. What did you change?
Cleaned memory cache to streamline the experiment. 
