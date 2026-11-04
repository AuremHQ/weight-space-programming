# The MIT Research, Explained Simply

Source: *"Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights"* — Rikiya Takehi, Jayden Teoh, Alex Zhang, Omar Khattab, MIT CSAIL, published 2 Nov 2026 as a blog post (not a peer-reviewed paper).

- Original blog: https://rikiyat.github.io/blog/weight-space-programming/
- Full experiment logs (prompts, replies, code, evaluations): https://github.com/RikiyaT/lm_understand-logs
- The edited Qwen model is on Hugging Face, linked from the blog.

This page summarizes the research. It is not the research itself, and nothing here should be quoted as MIT's own words except where marked.

## The core idea in one paragraph

A neural network is a pile of numbers called weights. Until now, the only way to change those numbers was training: show data, measure error, nudge the numbers with gradient descent. Weight-space programming says: a frontier language model can treat those numbers as readable, editable material. Give it the weights plus a plain-English goal, and it can find a hidden trait or failure inside them, fix it, or even write a small network from scratch — without any training loop.

The team's own summary line: instead of training, models may treat raw weights like readable, editable code — diagnosing failures and patching them directly.

## The 8 experiments

### Reading weights (no running the network)

1. **Which game?** Four game-playing networks (Breakout, Pong, Pac-Man, Freeway) with identical architecture (MLP, 17,253 weights), trained with PPO. Astra received only the weights as text, no game names, no options — and named the right game for all four. It read things like where each input channel's first-layer weights concentrate (for Breakout, the agent's weights lie along the bottom row, a paddle).
2. **What shape?** Three diffusion models trained to generate a circle, a triangle, or a square (23,824 weights each). From weights alone, Astra named each shape correctly — and drew them all as outlines, ignoring that the real models generate filled shapes, which suggests it was reading weight geometry, not simulating the network.
3. **What language, which heads?** Three small character-level transformers (4 layers × 6 heads, 7.16M weights) trained on Python code, Japanese in romaji, or English Wikipedia. With input and output layers removed (so the network could not run on any text at all), Astra named the right language for all three, in both attempts, and found every copying (induction) head while naming no false ones.

### Fixing weights

4. **The stuck car.** A PPO network driving a car that must reverse, then jump a ramp — stuck at 0% success (local minimum). In a single try with no tools, Astra's edit (3.5% of weights) made the car succeed in all 1,024 episodes, by adding a small path that backs the car up before driving forward.
5. **Writing game networks from scratch.** For three MinAtar games, Astra wrote all weights as text in one reply — no data, no loss, no gradients, no tools, never playing before scoring. Breakout: Astra 517.4 vs PPO trained for 500M steps 25.3 (about 20x). Asterix: 53.2 vs 55.2 (nearly tied). Seaquest: 53.1 vs 88.9 (60%). Against PPO trained for 100M steps, Astra won all three.

### Scaling up (tools allowed, still no training)

6. **Writing Go.** A 9×9 Go network (1.48M weights, all starting at zero) written from scratch in a Python sandbox, with a budget of forward passes (each = up to 128 practice games). With 0 forward passes it won more than half its games against two smaller PPO networks; with 50 forward passes it beat an equal-sized PPO network trained on 20 million games.
7. **Which model is sycophantic?** Two Gemma 3 270M models fine-tuned identically, one honest, one trained to always agree with the user's suggested answer. From weights alone Astra could NOT tell them apart (all four attempts failed). Given only attention patterns on texts Astra wrote itself, it picked the sycophant in all three pairs — middle-layer heads in the sycophant looked at the user's suggested answer instead of the question's numbers.

### The headline: a real language model

8. **Making Qwen3.5-0.8B better at math.** Qwen (24 layers, 0.8B weights) had a known flaw, warned about on its own model card: it falls into thinking loops, repeating text until it hits the 32,768-token limit (85 of its first 96 practice answers ran to the limit). Astra got a Python kernel over the weights, practice problems (not test sets), and a budget of forward passes. It traced the loop to one direction written by the MLPs of layers 15 and 19, and subtracted it from 64 output columns in each — 131,072 weights, 0.016% of the model.

| Qwen3.5-0.8B | MATH-500 | GSM8K | IFEval | Hit token limit |
|---|---|---|---|---|
| Base | 7.9% | 13.8% | 21.0% | 93.0% |
| Edited, 10 forward passes | 12.3% | 25.7% | 49.0% | 87.2% |
| Edited, 25 forward passes | 14.1% | 25.1% | 39.9% | 86.4% |
| Edited, 50 forward passes | **48.1%** | **67.1%** | **62.3%** | **33.1%** |

Note the budget dependence: the big jump needed 50 forward passes. At 10 and 25, the loops barely moved.

**The control that keeps everyone honest:** on Gemma 3 270M, which rarely loops (8.5% of answers hit its token limit vs Qwen's 93%), Astra could NOT improve it. Its failures were ordinary errors — wrong arithmetic, invalid algebra steps — with no single component to switch off. MATH-500 stayed around 6.8–7.7%. The authors' own conclusion: editing removes an obstacle hiding an ability; it cannot add an ability the model lacks.

## Why this could matter (the authors' vision)

- **Auditing without running:** find hidden problems or backdoors directly from weights, instead of only catching them with test sets or after deployment.
- **Cheaper optimization:** one-shot model creation instead of curating datasets and running full training loops.
- **Zero-data steering and unlearning:** add or remove a capability by prompt, touching only a few weights, avoiding the catastrophic forgetting fine-tuning often causes.
- **True model merging:** graft a circuit (e.g. a reasoning module) from one model into a different architecture, instead of today's weight averaging, which only works for identical architectures.
- **Recursive self-improvement:** a model inspecting, debugging, and improving its own weights, or a successor's.

The team says it is now building a benchmark and harness to turn what currently looks like "black magic" into a rigorous, reproducible engineering discipline.

## Where it stands — the honest limits

- **One editor model.** Everything above was done with GPT-6 Astra. Whether other frontier models can do this equally well is unproven.
- **Small models only.** The largest real model edited was 0.8B parameters. Nothing here shows this works on large production models.
- **Not peer-reviewed, not yet replicated.** This is a blog post with public logs. Independent replication is still pending.
- **A prompt-hint controversy.** A researcher noticed the Qwen prompt contained a hint about repetition (added by Claude Code while expanding the prompt, without further human feedback). The team's reply: a no-hint run at 25 passes also scored 14.1%, matching the hinted run. Treat the headline number as budget-dependent, not magic.
- **Context and harness limits.** The authors name context length and harness efficiency as the current bottlenecks.
- **Prior art exists.** Targeted weight editing is not brand new: ROME (NeurIPS 2022) edited factual associations in transformers, and MEMIT (ICLR 2023) extended that to thousands of associations at once. What is new here is a general-purpose model doing it natively, without being specially trained for the task.

## Relation to this repo

This repo does not reproduce MIT's experiments. It packages the method as a skill (see SKILL.md) for your own small open-weight models, with the same discipline the paper implies: diagnose a specific failure, make the smallest edit that fixes it, and let evaluation — not enthusiasm — decide whether it ships.
