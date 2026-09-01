# FAQ

**Which backend should I use, TRL or Unsloth?**
Unsloth gives faster training and lower memory use on supported model/GPU
combinations; TRL is the more general, broadly compatible default. See
[Backend Selection](../getting-started/backend-selection.md) and the
[Backend Comparison](../backends/comparison.md) for the full trade-offs.

**Does AlignTune work with my model?**
Check the [Unsloth Compatibility](../unsloth_compatibility.md) page and the
[Backend Support Matrix](../compatibility/backend-matrix.md) for known-good
and known-broken combinations.

**What algorithms are supported?**
13+ RL algorithms (DPO, PPO, GRPO, GSPO, DAPO, PACE, ORPO, SPIN, and more),
all documented in the [Algorithm Zoo](../algorithms/overview.md).

**Where do I report a bug or ask a usage question?**
See [Support](https://github.com/Lexsi-Labs/aligntune/blob/main/SUPPORT.md) ,
GitHub Discussions for questions, GitHub Issues for bugs, Discord for chat.

**How do I get help with training that isn't converging or is erroring out?**
Start with [Troubleshooting](../user-guide/troubleshooting.md); it covers the
most common SFT/RL failure modes before you need to open an issue.

**How do I cite AlignTune?**
See the [Cite](../index.md#cite) section on the home page, or the
[`CITATION.cff`](https://github.com/Lexsi-Labs/aligntune/blob/main/CITATION.cff)
file at the repository root.

**Is AlignTune free to use?**
It's free for research, evaluation, education, and non-revenue-generating use
under the Lexsi Labs Source Available License (LSAL) v1.1 — not an OSI open-source
license. Commercial use (selling, hosting as a paid/SaaS product, or embedding
in revenue-generating software) requires a separate commercial license from
Lexsi Labs. See [LICENSE.md](https://github.com/Lexsi-Labs/aligntune/blob/main/LICENSE.md),
or contact support@lexsi.ai for commercial licensing.
