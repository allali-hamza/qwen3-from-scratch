# Qwen3 from Scratch

A PyTorch implementation of the **Qwen3-1.7B** architecture, built from scratch.

The goal of this project is to understand how a modern large language model is assembled: from normalization and positional encoding to grouped-query attention, decoder blocks, and next-token prediction.

## What is implemented

- RMSNorm
- Rotary Positional Embeddings (RoPE)
- SwiGLU feed-forward network
- Grouped Query Attention (GQA)
- QK-Norm
- Causal attention masking
- Decoder blocks with residual connections
- Transformer backbone
- Causal language model with tied input and output embeddings
- Greedy autoregressive generation

## Why use pretrained weights?

The model architecture is implemented locally in this repository. The official Qwen3-1.7B weights are loaded only to verify that this implementation can run a real pretrained model.

The loader reads the official `safetensors` files, maps Qwen parameter names to this project’s module names, and checks that every model parameter is loaded.

## Installation

This project requires Python 3.12+ and uses `uv`.

```bash
uv sync
```

## Run the tests

```bash
uv run pytest -q
```

The test suite covers individual model components, weight loading with a local checkpoint, and autoregressive generation behavior.

## Run inference

Download the Qwen3-1.7B checkpoint and tokenizer once:

```bash
uv run hf download Qwen/Qwen3-1.7B
```

Then run the local implementation:

```bash
uv run inference_test.py
```

`inference_test.py` loads the pretrained weights into the model defined in this repository and generates text for several prompts.

## Project structure

```text
src/qwen3_from_scratch/
├── layers/
│   ├── attention.py
│   ├── decoder.py
│   ├── feed_forward.py
│   ├── normalization.py
│   └── rotary_embedding.py
├── generation.py
├── model.py
└── weights.py
```
