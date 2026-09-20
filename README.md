# TinyStories GPT — A From-Scratch Small Language Model

A ~25-30M parameter GPT-style transformer, trained entirely from scratch on the [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories), built as an end-to-end learning project covering tokenization, model architecture, training, checkpointing, and inference.

## Overview

This project implements a decoder-only transformer (GPT-style) trained to generate short, coherent children's stories. It was trained on consumer-grade hardware (tokenization) and a free Kaggle T4 GPU (training), demonstrating that a small, functional language model can be built without expensive infrastructure.

## Model Details

- **Architecture:** Decoder-only Transformer (GPT-style), custom implementation
- **Parameters:** ~25-30M
- **Layers:** 6
- **Attention heads:** 6
- **Embedding dimension:** 384
- **Context length (block size):** 256 tokens
- **Vocabulary:** GPT-2 BPE tokenizer (50,257 tokens)
- **Training data:** TinyStories dataset (~470M tokens)
- **Training steps:** 20,000

- 
**Note:** `train.bin`, `val.bin`, and trained model weights are not included in this repository due to size (see below).

## How It Works

1. **Tokenization** (`tokenize_data.py`): Downloads TinyStories via Hugging Face `datasets`, tokenizes it using GPT-2's BPE tokenizer, and saves the result as compact binary files (`train.bin`, `val.bin`) using `uint16` token IDs for efficient storage and fast memory-mapped loading.

2. **Training** (`GPT.py`): Implements a GPT-style transformer from scratch (multi-head self-attention, causal masking, feedforward blocks, layer normalization). Trains using next-token prediction with cross-entropy loss, AdamW optimizer, and periodic checkpointing (model + optimizer state) to support resuming across sessions.

3. **Inference** (`GPT.py`): Loads trained weights and generates text via autoregressive sampling, with temperature and top-k controls for coherence tuning.

## Setup & Usage

### Requirements
```bash
pip install torch transformers datasets numpy
```

### 1. Tokenize the dataset
```bash
python tokenize_data.py
```
This downloads TinyStories and produces `train.bin` and `val.bin` (~1GB total).

### 2. Train the model
Training was run on Kaggle (free T4 GPU) — see `GPT.py` for the training loop. Adjust `max_steps`, `batch_size`, and model config as needed for your hardware.

### 3. Generate text
```python
from GPT import TinyGPT, generate
import torch
from transformers import AutoTokenizer

model = TinyGPT(vocab_size=50257, n_embd=384, n_head=6, n_layer=6, block_size=256)
model.load_state_dict(torch.load("path/to/tinygpt_final_weights.pt", map_location="cpu"))
model.eval()

tokenizer = AutoTokenizer.from_pretrained("gpt2")
print(generate(model, tokenizer, "Once upon a time, there was a"))
```

## Sample Output

> Once upon a time, there was a little girl named Lily. She loved to eat cake every day. One day, her mom made her a sweet cake for dinner. Lily was happy...

## Known Limitations

- Trained on a narrow, simplified dataset (TinyStories) — coherent for short story continuation only, not general-purpose Q&A or instruction-following
- Occasional semantic inconsistencies typical of small-scale (~25-30M param) models
- Not intended as a production assistant — this is an educational/research project demonstrating the full LLM training pipeline at small scale

## Model Weights

Trained weights (~100-120MB) are hosted separately due to GitHub file size limits: [add your Hugging Face / Kaggle Dataset / Google Drive link here]

## Acknowledgments

- [TinyStories dataset](https://arxiv.org/abs/2305.07759) (Eldan & Li, Microsoft Research)
- Architecture inspired by [nanoGPT](https://github.com/karpathy/nanoGPT)
- Trained using free GPU access via [Kaggle Notebooks](https://kaggle.com)

## License

[Choose one — MIT is common for learning/research projects like this]
- **Hardware:** Single NVIDIA T4 GPU (Kaggle free tier)

## Project Structure
