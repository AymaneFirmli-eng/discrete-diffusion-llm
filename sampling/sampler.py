import torch
from tokenizers import Tokenizer
from models.transformer import DiffusionTransformer

@torch.no_grad()
def generate_text(
    model: DiffusionTransformer,
    tokenizer: Tokenizer,
    seq_len: int = 128,
    num_steps: int = 20
):
    model.eval()
    device = next(model.parameters()).device
    
    mask_token_id = tokenizer.token_to_id("<mask>")
    
    # 1. Start with a sequence full of [MASK] tokens
    current_ids = torch.full((1, seq_len), mask_token_id, dtype=torch.long, device=device)
    
    # 2. Iterative progressive unmasking loop from t = 1.0 down to 0.0
    timesteps = torch.linspace(1.0, 0.0, num_steps, device=device)
    
    for i in range(num_steps - 1):
        t = timesteps[i]
        t_tensor = torch.tensor([t], device=device)
        
        # Predict logits for all positions simultaneously
        logits = model(current_ids, t_tensor) # [1, seq_len, vocab_size]
        probs = torch.softmax(logits, dim=-1)
        
        # Get confidence scores and predicted token ids
        confidences, predicted_ids = torch.max(probs, dim=-1) # [1, seq_len]
        
        # Determine how many tokens to unmask based on current timestep schedule
        next_t = timesteps[i + 1]
        mask_ratio = next_t
        
        # Identify which tokens are currently masked
        is_masked = (current_ids == mask_token_id)
        
        if is_masked.sum() > 0:
            # Select lowest confidence tokens to keep masked, unmask the high confidence ones
            num_to_mask = int(seq_len * mask_ratio)
            
            # Sort confidences of currently masked tokens
            masked_confidences = confidences.clone()
            masked_confidences[~is_masked] = float('inf') # ignore already unmasked tokens
            
            _, sorted_indices = torch.sort(masked_confidences, dim=1)
            
            # Create new token state
            new_ids = current_ids.clone()
            new_ids[0] = predicted_ids[0]
            
            # Re-mask the lowest confidence positions
            if num_to_mask > 0:
                tokens_to_remask = sorted_indices[0, :num_to_mask]
                new_ids[0, tokens_to_remask] = mask_token_id
                
            current_ids = new_ids

    # Final decode to text string
    final_ids = current_ids[0].tolist()
    text = tokenizer.decode(final_ids)
    return text