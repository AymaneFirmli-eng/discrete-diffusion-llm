import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

def load_and_tokenize_dataset(batch_size=32, max_length=256):
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    
    print("Loading tokenizer and dataset...")
    tokenizer = Tokenizer.from_file("tokenizer/saved_tokenizer/tokenizer.json")
    
    # Get special token IDs
    mask_token_id = tokenizer.token_to_id("<mask>")
    pad_token_id = tokenizer.token_to_id("<pad>")
    
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    
    def collate_fn(batch):
        texts = [item["text"] for item in batch]
        encoded = tokenizer.encode_batch(texts)
        
        input_ids = []
        for enc in encoded:
            ids = enc.ids[:max_length]
            # Pad if shorter than max_length
            if len(ids) < max_length:
                ids = ids + [pad_token_id] * (max_length - len(ids))
            input_ids.append(ids)
            
        return torch.tensor(input_ids, dtype=torch.long)

    # Simple iterable dataset wrapper for streaming
    class StreamDataset(torch.utils.data.IterableDataset):
        def __init__(self, ds):
            self.ds = ds
        def __iter__(self):
            for item in self.ds:
                yield item

    train_loader = DataLoader(
        StreamDataset(dataset),
        batch_size=batch_size,
        collate_fn=collate_fn
    )
    return train_loader, mask_token_id, pad_token_id


def compute_diffusion_loss(model, input_ids, mask_token_id, pad_token_id):
    """
    Implements Masked Discrete Diffusion corruption and cross-entropy loss.
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    
    # 1. Sample random diffusion timestep t for each item in batch (ranging from 0.0 to 1.0)
    timesteps = torch.rand(batch_size, device=device)
    
    # 2. Determine masking probability based on timestep (higher t = more masking)
    # Using a cosine schedule or linear masking ratio correlated with t
    mask_probs = timesteps.unsqueeze(1).expand(-1, seq_len)
    
    # Don't mask padding tokens
    is_pad = (input_ids == pad_token_id)
    
    # 3. Create random mask tensor
    random_vals = torch.rand(batch_size, seq_len, device=device)
    is_masked = (random_vals < mask_probs) & (~is_pad)
    
    # 4. Corrupt input: replace masked positions with [MASK] token ID
    corrupted_input_ids = input_ids.clone()
    corrupted_input_ids[is_masked] = mask_token_id
    
    # 5. Forward pass through transformer
    logits = model(corrupted_input_ids, timesteps) # Shape: [batch_size, seq_len, vocab_size]
    
    # 6. Compute Cross-Entropy Loss ONLY on the masked tokens
    # If no tokens were masked in a row, avoid NaN by fallback
    if not is_masked.any():
        return torch.tensor(0.0, device=device, requires_grad=True)
        
    loss = F.cross_entropy(
        logits[is_masked],
        input_ids[is_masked],
        reduction='mean'
    )
    
    return loss