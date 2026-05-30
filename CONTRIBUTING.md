# Contributing

This repository is a research code release for SPADER. Contributions that improve reproducibility, fix bugs in the released training/evaluation path, or clarify documentation are welcome.

Before opening a pull request:

- keep changes focused on the released Qwen3-8B/Llama3-8B GRPO and SPADER workflows,
- avoid adding large generated data, model weights, checkpoints, logs, or retrieval indexes,
- run the relevant smoke checks for the code you changed,
- update `README.md` when user-facing commands change.

The core package keeps the `verl` module name for compatibility with the underlying training framework.
