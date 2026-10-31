# AI use summary

**Tool:** Claude Code (Anthropic's Claude), used as a coding assistant.

**How it helped:** it wrote the Python scripts, tests, and browser demo, drafted the README, ran
the training, tuning, and checks, and helped with research on parameter matching, overfitting,
and residual connections.

**My part:** I made the design decisions and reviewed and approved every change.

**Checks:** every number in the README comes from the scripts' saved outputs in `results/`;
tests and lint pass in CI, and a parity check confirms the demo matches PyTorch.
