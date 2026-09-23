import torch
import torch.nn as nn
from models.embeddings import DiffusionEmbedding

class DiffusionTransformerBlock(nn.Module):
    """
    Standard Transformer Encoder block with bidirectional self-attention 
    and pre-layer normalization.
    """
    def __init__(self, hidden_dim: int, num_heads: int, intermediate_dim: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        # SwiGLU / Feed-Forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Self-Attention (Bidirectional / No Causal Mask)
        norm_x = self.ln1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x, need_weights=False)
        x = x + attn_out
        
        # 2. Feed-Forward Network
        x = x + self.ffn(self.ln2(x))
        return x


class DiffusionTransformer(nn.Module):
    """
    The ~45M Parameter Diffusion Language Model Backbone.
    """
    def __init__(
        self,
        vocab_size: int = 16384,
        hidden_dim: int = 768,      # Hidden size
        num_layers: int = 12,       # Number of transformer blocks
        num_heads: int = 12,        # Attention heads
        intermediate_dim: int = 3072, # FFN dimension
        max_seq_len: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embedding = DiffusionEmbedding(vocab_size, hidden_dim, max_seq_len)
        
        self.blocks = nn.ModuleList([
            DiffusionTransformerBlock(hidden_dim, num_heads, intermediate_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.ln_out = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size, bias=False)
        
        # Weight tying for efficiency and cleaner learning
        self.head.weight = self.embedding.token_embedding.weight

    def forward(self, input_ids: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        # 1. Get combined embeddings (Tokens + Positions + Timesteps)
        x = self.embedding(input_ids, timesteps)
        
        # 2. Pass through bidirectional transformer blocks
        for block in self.blocks:
            x = block(x)
            
        # 3. Final norm and projection back to vocabulary logits
        x = self.ln_out(x)
        logits = self.head(x)
        
        return logits