import os
from datasets import load_dataset
from tokenizers import Tokenizer, trainers, pre_tokenizers, processors
from tokenizers.models import BPE

def train_custom_tokenizer():
    print("Loading TinyStories dataset locally...")
    # Streaming a slice of TinyStories to train our custom BPE tokenizer
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    
    def batch_iterator(batch_size=1000):
        batch = []
        for i, example in enumerate(dataset):
            batch.append(example["text"])
            if len(batch) == batch_size:
                yield batch
                batch = []
            if i >= 50000:  # 50k stories for a solid vocabulary base
                break
        if batch:
            yield batch

    print("Initializing Byte-Level BPE Tokenizer...")
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Special tokens required for our diffusion model and chat formatting
    special_tokens = ["<pad>", "<mask>", "<bos>", "<eos>", "<unk>"]
    
    trainer = trainers.BpeTrainer(
        vocab_size=16384,
        special_tokens=special_tokens,
        min_frequency=2
    )

    print("Training tokenizer...")
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
    tokenizer.post_processor = processors.ByteLevel(use_alphabet_prefix=False)

    os.makedirs("tokenizer/saved_tokenizer", exist_ok=True)
    tokenizer.save("tokenizer/saved_tokenizer/tokenizer.json")
    print("Tokenizer successfully trained and saved to tokenizer/saved_tokenizer/tokenizer.json!")

if __name__ == "__main__":
    train_custom_tokenizer()