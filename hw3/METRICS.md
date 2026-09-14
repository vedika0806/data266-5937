
## Part 2 — Causal Masking Comparison

- **Unmasked model** final training loss: 0.0833
- **Masked model** final training loss: 0.1497
In unmasked self-attention, tokens attend to both past and future positions. Early tokens like "neural" attend to later tokens like "layers" or "entirely". That is the heatmap shows correlation coefficient spread all over the grid.
In causal masked self attention (like that in autoregressive models), the upper triangle is exactly zero, because each token attends to itself and the prior tokens. This matches the decoder self-attention behavior described in
Section 3.2.3 of *Attention Is All You Need*, which prevents positions from attending to subsequent positions to preserve the autoregressive property.
The loss trajectory of the causal masked self attention has fluctuations until 100 epochs after which it settles. The same is not the case with the unmasked self attention model because it trains on fixed, predictable text targets, allowing the model to quickly memorize patterns and settle early.Unmasked attention relies on random, dynamic masking that changes every epoch, presenting a moving target that forces longer training fluctuations.

## Part 1 — Prompt Engineering Technique Comparison

- **Zero-Shot**: Fast, no examples needed, but occasionally skips reasoning steps and
  can misfire on tasks requiring multi-step logic.
- **Few-Shot**: Anchors output format/style via examples; improved consistency on the
  logic questions vs zero-shot, but quality depends heavily on example choice.
- **Chain-of-Thought (explicit)**: Produces visible intermediate steps, catches
  arithmetic/logic errors that zero-shot missed, at the cost of longer output.
- **Zero-Shot CoT**: Nearly matched explicit CoT quality just by appending
  "Let's think step by step," with less prompt-engineering effort.
- **Meta-Prompting**: Model articulates a strategy before solving; useful for
  ambiguous/multi-part tasks, but adds overhead for simple problems.
- **Tree of Thoughts**: Generated multiple candidate reasoning paths and picked the
  best; most robust on the constraint-satisfaction (chickens/cows) problem, but
  most expensive (multiple LLM calls).

**Overall finding**: reasoning-heavy tasks (arithmetic word problems, multi-step
logic) benefited most from CoT/ToT; simple factual asks did fine with Zero-Shot.
