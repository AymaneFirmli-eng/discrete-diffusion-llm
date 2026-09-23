import math
import torch
import torch.nn as nn

class TimestepEmbedding(nn.Module):
    """
    Sinusoidal time-step embeddings similar to Diffusion models (like DDPM/Image Diffusion),
    projected to match the model hidden dimension so it can be added to token embeddings.
    """
    def __init__(self, hidden_dim: int, max_period: int = 10000):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_period = max_period
        
        # MLP projection for time steps
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        # timesteps shape: [batch_size] or scalar
        half_dim = self.hidden_dim // 2
        frequencies = torch.exp(
            -math.log(self.max_period) * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) / half_dim
        )
        args = timesteps[:, None].float() * frequencies[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if self.hidden_dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            
        return self.mlp(embedding)


class DiffusionEmbedding(nn.Module):
    """
    Combines Token Embeddings, Positional Embeddings, and Timestep Embeddings.
    """
    def __init__(self, vocab_size: int, hidden_dim: int, max_seq_len: int = 512):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
        self.timestep_embedding = TimestepEmbedding(hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, input_ids: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        
        # 1. Token Embeddings
        tok_emb = self.token_embedding(input_ids) * math.sqrt(self.hidden_dim)
        
        # 2. Positional Embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions)
        
        # 3. Timestep Embeddings (broadcasted across sequence length)
        time_emb = self.timestep_embedding(timesteps).unsqueeze(1).expand(-1, seq_len, -1)
        
        # Sum them all together
        return tok_emb + pos_emb + time_emb