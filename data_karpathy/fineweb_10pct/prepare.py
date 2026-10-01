"""
Prepare the FineWeb-Edu dataset (10% sample) for training.
Downloads from HuggingFace, tokenizes with GPT-2 BPE, saves as .bin files.
"""

import os
import numpy as np
import tiktoken
from datasets import load_dataset

# Create data directory
data_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(data_dir, exist_ok=True)

# Initialize GPT-2 tokenizer
enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab

print(f"Downloading FineWeb-Edu dataset (10% sample)...")
dataset = load_dataset("HuggingFaceFW/fineweb-edu", "default", split="train", trust_remote_code=True)

# Use only 10% of the data
total_samples = len(dataset)
samples_10pct = max(1, total_samples // 10)
print(f"Total samples: {total_samples:,}")
print(f"Using 10%: {samples_10pct:,} samples")

# Take every 10th sample to get 10% (more efficient than random sampling)
dataset = dataset.select(range(0, total_samples, 10))
print(f"Dataset size after 10% sampling: {len(dataset):,}")

# Split into train/val (90/10 of the 10%)
val_size = max(1, len(dataset) // 10)
split_dataset = dataset.train_test_split(test_size=val_size, seed=42)
train_dataset = split_dataset['train']
val_dataset = split_dataset['test']

print(f"Train samples: {len(train_dataset):,}")
print(f"Val samples: {len(val_dataset):,}")

def tokenize_and_save(dataset, output_file):
    """Tokenize dataset and save as binary file."""
    print(f"Tokenizing {output_file}...")
    
    all_tokens = []
    for idx, sample in enumerate(dataset):
        if idx % 1000 == 0:
            print(f"  Processing sample {idx}/{len(dataset)}")
        
        text = sample['text']
        tokens = enc.encode_ordinary(text)
        all_tokens.extend(tokens)
    
    # Convert to numpy array and save
    all_tokens = np.array(all_tokens, dtype=np.uint16)
    with open(output_file, 'wb') as f:
        all_tokens.tofile(f)
    
    print(f"  Saved: {output_file} ({len(all_tokens):,} tokens)")
    return len(all_tokens)

# Tokenize and save train/val
train_tokens = tokenize_and_save(train_dataset, os.path.join(data_dir, 'train.bin'))
val_tokens = tokenize_and_save(val_dataset, os.path.join(data_dir, 'val.bin'))

# Save metadata
import pickle
meta = {
    'vocab_size': vocab_size,
    'train_tokens': train_tokens,
    'val_tokens': val_tokens,
}
with open(os.path.join(data_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print(f"\n✅ Data preparation complete!")
print(f"Train: {train_tokens:,} tokens")
print(f"Val: {val_tokens:,} tokens")
print(f"Vocab size: {vocab_size}")
