# Implement a LLaMA-3 from scratch

This repository implements a single-layer version of LLaMA-3, primarily based on the repository
at [llama-from-scratch](https://github.com/naklecha/llama3-from-scratch) and [official-repo](https://github.com/meta-llama/llama3/tree/main/llama). The key differences are:

- The current repository uses python files instead of jupyter notebooks (fundamentally similar), which may provide better readability and a more streamlines experience.

- The current repository provides Chinese comments to help Chinese learner.

Limitations:

- Our project does not use `fairscale` mentioned in the [official-repo](https://github.com/meta-llama/llama3/tree/main/llama), so any parallel strategies are not considered.

- The concept of `kv_cache` is not introduced, which may lead to higher memory requirements.

- It does not support batch processing.
