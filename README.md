<div align="center">

# SPADER

**Step-wise Peer Advantage with Diversity-Aware Exploration Rewards for Multi-Answer Question Answering**

<p>
  <a href="https://huggingface.co/KhanCold/qwen3-8b-spader"><img alt="Qwen3-8B checkpoint" src="https://img.shields.io/badge/HF-Qwen3--8B--SPADER-ffcc4d?logo=huggingface&logoColor=black"></a>
  <a href="https://huggingface.co/KhanCold/llama3-8b-spader"><img alt="Llama checkpoint" src="https://img.shields.io/badge/HF-Llama--3.1--8B--SPADER-ffcc4d?logo=huggingface&logoColor=black"></a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776ab?logo=python&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/License-Apache--2.0-2f6f5e">
</p>

<p>
  <a href="#released-models">Models</a> |
  <a href="#data">Data</a> |
  <a href="#environment">Environment</a> |
  <a href="#retrieval">Retrieval</a> |
  <a href="#spader-rl-training">Training</a> |
  <a href="#evaluation">Evaluation</a>
</p>

</div>

## Abstract

Large language models are increasingly deployed as tool-augmented agents to acquire information beyond parametric knowledge.
While recent work has improved long-horizon tool-use reasoning, most approaches focus on tasks with a single correct answer.
In contrast, many real-world queries require discovering a comprehensive set of valid answers, a setting known as Multi-Answer QA.
This setting raises two challenges: fine-grained credit assignment over long search trajectories and reward alignment for sustained exploration beyond easy high-frequency entities.
We propose **SPADER**, a reinforcement learning framework for long-horizon tool use in Multi-Answer QA.
SPADER includes Step-wise Peer Advantage (SPA), a critic-free step-level credit assignment mechanism that aligns parallel trajectories by decision step and estimates advantages from peer returns.
It also includes a diversity-aware exploration reward that promotes long-tail entity discovery by upweighting rare findings and downweighting redundant ones.
Experiments on QAMPARI, Mintaka, WebQSP, and QUEST show that SPADER generally improves recall and overall F1 over prompting-based agents, outcome-supervised RL methods, and recent step-level supervision approaches.
Our code and model weights are available at [https://github.com/KhanCold/spader](https://github.com/KhanCold/spader).

SPADER trains tool-augmented search agents for multi-answer QA. The repository keeps the original VERL training stack and adds a SPADER reward manager plus a masked step-level GRPO advantage estimator.

## Overview

| Component               | Scope                                                     |
| ----------------------- | --------------------------------------------------------- |
| Retrieval environment   | Local HTTP service for Wikipedia-based search             |
| Training implementation | VERL-based SPADER with step-wise peer advantage (**SPA**) |
| Reward design           | Diversity-aware exploration reward manager                |
| Evaluation              | Lightweight scripts for released model variants           |
| Checkpoints             | Qwen3-8B and Llama-3.1-8B SPADER releases                 |

## Released Models

| Model              | Hugging Face Repo           | Download                                                                                        |
| ------------------ | --------------------------- | ----------------------------------------------------------------------------------------------- |
| `qwen3-8b-spader`  | [`KhanCold/qwen3-8b-spader`](https://huggingface.co/KhanCold/qwen3-8b-spader)  | `huggingface-cli download KhanCold/qwen3-8b-spader --local-dir spader/models/qwen3-8b-spader`   |
| `llama3-8b-spader` | [`KhanCold/llama3-8b-spader`](https://huggingface.co/KhanCold/llama3-8b-spader) | `huggingface-cli download KhanCold/llama3-8b-spader --local-dir spader/models/llama3-8b-spader` |

## Repository Layout

```text
examples/sglang_multiturn/
  config/                         # QAMPARI training configs
  qampari/                        # training launch scripts
  search_r1_like/local_dense_retriever/
                                  # local retrieval server
run_test/
  inference_scripts/              # vLLM serving scripts
  run_sh/                         # evaluation launch scripts
  run_eval.py                     # evaluation entrypoint
verl/                             # VERL training framework with SPADER additions
```

## Data

Place training files anywhere on disk and pass them with `TRAIN_DATA` and `VAL_DATA`.
Training uses VERL-style `.parquet` files, where each sample should include:

| Field          | Type                | Description                                      |
| -------------- | ------------------- | ------------------------------------------------ |
| `prompt`       | list of messages    | Chat messages consumed by the rollout agent      |
| `data_source`  | string              | Dataset or task name used by the reward function |
| `reward_model` | dict                | Contains `ground_truth` for scoring              |
| `extra_info`   | dict, optional      | Optional metadata such as `index`                |

For Multi-Answer QA, `reward_model.ground_truth` should contain a list of answer entities with aliases:

```json
{
  "ground_truth": [
    ["Entity A", "Alias A"],
    ["Entity B"]
  ]
}
```

Evaluation uses `.jsonl` files, one JSON object per line. Each object should include:

| Field          | Type             | Description                                      |
| -------------- | ---------------- | ------------------------------------------------ |
| `question_text` or `question` | string | Input question                         |
| `ground_truth` | list of lists    | Gold entities and aliases                        |
| `data_source`  | string, optional | Dataset or task name                             |

Place evaluation files under `data/evaluation` by default, or set `EVAL_DATA_DIR` to another directory.
Named evaluation datasets are resolved from `EVAL_DATA_DIR`.
For example, `--dataset qampari` reads `data/evaluation/qampari_test.jsonl`.
You can also pass an explicit `.jsonl` path with `--dataset /path/to/file.jsonl`.

## Environment

Create the main training/runtime environment:

```bash
conda create --name verl-sglang python=3.12
conda activate verl-sglang
pip install -e ".[sglang,vllm]"
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
pip install cachetools
```

If your CUDA/PyTorch build needs FlashAttention installed manually, install the wheel matching your CUDA, Python, and PyTorch versions. The experiments used a CUDA 12 build.

Create the retrieval environment:

```bash
conda env create -f retriever.yml
conda activate retriever
pip install torch==2.8.0 torchaudio==2.8.0 torchvision --index-url https://download.pytorch.org/whl/cu128
```

## Base Models for training

Download the base models to `models/`:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3-8B",
    local_dir="models/Qwen3-8B",
    allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*", "*.py"],
    ignore_patterns=["*.bin", "*.h5", "*.ot", "*.msgpack"],
)

snapshot_download(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    local_dir="models/Llama-3.1-8B-Instruct",
    allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.py"],
    ignore_patterns=["*.bin", "*.h5", "*.ot", "*.msgpack"],
)
PY
```

Training scripts default to the paths above. Override `BASE_MODEL` if your checkpoints are stored elsewhere.

## Retrieval

SPADER uses an HTTP retrieval service at `http://127.0.0.1:8000/retrieve`.

For QAMPARI, use the Wikipedia 2021 index. Build it from the chunked dump:

```bash
mkdir -p data/QAMPARI_wikipedia_2021
wget "https://aggreg-qa.s3.amazonaws.com/chunked_wikipedia.tar.gz" \
  -O data/QAMPARI_wikipedia_2021/chunked_wikipedia.tar.gz
bash examples/sglang_multiturn/search_r1_like/local_dense_retriever/build_qampari_index.sh
```

Or download the prebuilt index:

```bash
mkdir -p data
wget "https://huggingface.co/datasets/KhanCold/QAMPARI_wikipedia_2021/resolve/main/QAMPARI_wikipedia_2021.tar" \
  -O data/QAMPARI_wikipedia_2021.tar
tar -xf data/QAMPARI_wikipedia_2021.tar -C data
```

Optional BM25 + reranker mode:

```bash
huggingface-cli download BAAI/bge-reranker-v2-m3 \
  --local-dir models/bge-reranker-v2-m3 \
  --local-dir-use-symlinks False

conda activate retriever
bash examples/sglang_multiturn/search_r1_like/local_dense_retriever/start_bm25_rerank_server.sh
```

Smoke test:

```bash
curl -X POST "http://localhost:8000/retrieve" \
  -H "Content-Type: application/json" \
  -d '{"queries": ["Mel Brooks directed and produced films"], "topk": 3}'
```

## SPADER RL Training

Run commands from the repository root. All scripts accept normal Hydra overrides after the script name.

| Setting             | Command                                                          |
| ------------------- | ---------------------------------------------------------------- |
| Qwen3-8B GRPO       | `bash examples/sglang_multiturn/qampari/run_qwen3-8b_grpo.sh`    |
| Qwen3-8B SPADER     | `bash examples/sglang_multiturn/qampari/run_qwen3-8b_spader.sh`  |
| Llama-3.1-8B GRPO   | `bash examples/sglang_multiturn/qampari/run_llama3-8b_grpo.sh`   |
| Llama-3.1-8B SPADER | `bash examples/sglang_multiturn/qampari/run_llama3-8b_spader.sh` |

Useful overrides:

```bash
BASE_MODEL=models/Qwen3-8B \
TRAIN_DATA=/path/to/train.parquet \
VAL_DATA=/path/to/val.parquet \
OUTPUT_ROOT=output \
bash examples/sglang_multiturn/qampari/run_qwen3-8b_spader.sh \
  trainer.n_gpus_per_node=8 \
  trainer.save_freq=100
```

`TRAIN_DATA` and `VAL_DATA` are required by the launch scripts.

SPADER-specific code paths:

| Module              | Path                                                                      |
| ------------------- | ------------------------------------------------------------------------- |
| Reward manager      | `verl/workers/reward_manager/spader_step.py`, registered as `spader_step` |
| Advantage estimator | `masked_step_grpo` in `verl/trainer/ppo/core_algos.py`                    |
| Configs             | `examples/sglang_multiturn/config/qampari_*_spader.yaml`                  |

## Evaluation

First merge an FSDP actor checkpoint to Hugging Face format if needed:

```bash
python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir output/<run_name>/checkpoints/global_step_<step>/actor \
  --target_dir models/qwen3-8b-spader
```

Serve the model with vLLM:

```bash
bash run_test/inference_scripts/run_qwen3_8b_spader.sh
```

Then evaluate:

```bash
EVAL_DATA_DIR=/path/to/evaluation_data \
python run_test/run_eval.py \
  --dataset all \
  --model qwen3-8b-spader \
  --concurrency 20 \
  --analyze
```

Convenience scripts are available for all released variants:

```bash
bash run_test/run_sh/run_eval_qwen3_8b_grpo.sh
bash run_test/run_sh/run_eval_qwen3_8b_spader.sh
bash run_test/run_sh/run_eval_llama3_8b_grpo.sh
bash run_test/run_sh/run_eval_llama3_8b_spader.sh
```

Results are written to `run_test/results/<dataset>/`.

## Citation

```bibtex
@misc{shi2026spaderstepwisepeeradvantage,
      title={SPADER: Step-wise Peer Advantage with Diversity-Aware Exploration Rewards for Multi-Answer Question Answering}, 
      author={Qiming Shi and Zhaolu Kang and Yunfan Zhou and Di Weng and Yingcai Wu},
      year={2026},
      eprint={2606.00593},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2606.00593}, 
}
```
