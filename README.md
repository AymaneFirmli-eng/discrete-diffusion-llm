# 45M-Parameter Discrete Diffusion Language Model (dLLM) from Scratch

A custom implementation of a **Masked Discrete Diffusion Language Model** built from scratch using PyTorch and trained on the TinyStories dataset.

## Architecture Highlights
- **~45M Parameter Transformer Encoder** (12 layers, hidden dimension 768, 12 attention heads).
- **Custom Byte-Level BPE Tokenizer** trained from scratch.
- **Bidirectional (Unmasked) Attention** enabling tokens to leverage both left and right context.
- **Sinusoidal Timestep Embeddings** integrated into token representations.
- **Iterative Denoising Sampler** for progressive parallel unmasking.

## Project Structure
- `tokenizer/` - Custom BPE tokenizer training script.
- `models/` - Transformer backbone and timestep/positional embeddings.
- `training/` - Masked diffusion noise schedule and loss calculations.
- `sampling/` - Iterative confidence-based unmasking generation loop.