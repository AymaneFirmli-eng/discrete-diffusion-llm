import torch
from accelerate import Accelerator
from models.transformer import DiffusionTransformer
from training.loss import load_and_tokenize_dataset, compute_diffusion_loss

def main():
    # If running on Mac M1, PyTorch will automatically use MPS if available, 
    # but Accelerate handles device placement cleanly.
    accelerator = Accelerator(mixed_precision="no")
    
    print("Initializing Diffusion Transformer...")
    model = DiffusionTransformer(
        vocab_size=16384,
        hidden_dim=768,
        num_layers=12,
        num_heads=12,
        intermediate_dim=3072,
        max_seq_len=256
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    train_loader, mask_token_id, pad_token_id = load_and_tokenize_dataset(batch_size=8)
    
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    
    model.train()
    print("Starting training loop...")
    
    step = 0
    for batch in train_loader:
        batch = batch.to(accelerator.device)
        
        optimizer.zero_grad()
        loss = compute_diffusion_loss(model, batch, mask_token_id, pad_token_id)
        
        accelerator.backward(loss)
        optimizer.step()
        
        if step % 10 == 0:
            print(f"Step {step} | Loss: {loss.item():.4f}")
            
        if step > 0 and step % 1000 == 0:
            accelerator.save_state(f"saved_checkpoints/step_{step}")
            print(f"Checkpoint saved at step {step}!")
            
        step += 1
        if step >= 5000: # Training run limit for demo/learning
            break

if __name__ == "__main__":
    main()