
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
