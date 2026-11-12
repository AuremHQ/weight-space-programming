# Reel kit: make a 30-second wsp-kit video

You do not need to be an expert. The story is already good: "I fixed my AI's bug by editing 0.02% of its weights, and the gate said yes." Or, just as good: "The gate said no, and nothing shipped."

## 30-second script (vertical, 9:16)

| Time | On screen | Say |
|---|---|---|
| 0-3s | Text: "My small AI kept repeating itself." Show the looping output | "My local model loops when I ask for 60 animals." |
| 3-8s | Terminal: `wsp run --model ... --eval ...` | "One command. No retraining. No gradients." |
| 8-18s | Scroll the gate result | "A gate re-tests the same prompts. It needs plus 2 points, and at most minus 2 on anything else." |
| 18-26s | The result line, e.g. 37.78 -> 67.22 | "Failing score: 37.78 to 67.22. Under 1% of weights changed." |
| 26-30s | Text: "github.com/AuremHQ/weight-space-programming" | "If the gate says no, nothing ships. Link in bio." |

## Rules that keep it honest (and trusted)

Show your own numbers from your run. Do not quote ours as if they were yours. Say "small model" and "one failure". Do not say "fixes AI" or "makes it smarter". A refusal is a good reel: "I ran it, the gate refused, nothing shipped." Mention the kit is independent and inspired by MIT CSAIL research, not made by MIT.

## Hook ideas (first 2 seconds decide)

- "The gate said NO five times out of seven."
- "I edited 0.02% of a model's weights."
- "No retraining. No gradients. Just a gate."

## Hashtags

#opensource #LocalLLM #AI #MachineLearning #Qwen #buildinpublic

## After posting

Open a My run issue (Issues > New issue) with your numbers and link the reel.
