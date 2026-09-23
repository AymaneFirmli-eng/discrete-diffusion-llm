# 110M-Parameter Discrete Diffusion Language Model (dLLM) from Scratch

A custom implementation of a **Masked Discrete Diffusion Language Model** built from scratch using PyTorch and trained on a slice of the TinyStories dataset. 

Unlike traditional autoregressive language models (like GPT) that generate text strictly from left to right, this model implements **Masked Diffusion Language Modeling (MDLM)**, allowing it to iteratively generate and refine text in parallel across the entire sequence length.

## Architecture Highlights
- **~110M Parameter Transformer Encoder** (Configured with deep layers, hidden dimension 768, and 12 attention heads).
- **Custom Byte-Level BPE Tokenizer** trained from scratch, supporting specialized tokens (`[MASK]`, `[BOS]`, `[EOS]`, `[PAD]`).
- **Bidirectional (Unmasked) Attention** enabling every token to leverage both left and right context simultaneously during the forward pass.
- **Sinusoidal Timestep Embeddings** dynamically injected into token representations to inform the model of the current noise/diffusion level ($t$).
- **Iterative Denoising Sampler** featuring progressive unmasking and confidence-based token selection during inference.

## Technical Stack
- **Language**: Python 3.10+
- **Deep Learning**: PyTorch
- **Distributed & Scaling**: Hugging Face Accelerate
- **Data & Tokenization**: Hugging Face Datasets & Tokenizers

## Project Structure
```text
discrete-diffusion-llm/
├── tokenizer/
│   ├── train_tokenizer.py
│   └── saved_tokenizer/
│       └── tokenizer.json
├── models/
│   ├── __init__.py
│   ├── embeddings.py
│   └── transformer.py
├── training/
│   ├── __init__.py
│   └── loss.py
├── sampling/
│   ├── __init__.py
│   └── sampler.py
├── train.py
├── requirements.txt
└── README.md