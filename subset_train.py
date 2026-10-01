import argparse
import numpy as np
import os
import random

parser = argparse.ArgumentParser()
parser.add_argument("--percent", type=int, default=10) # default is 10% of dataset
args = parser.parse_args()

NUM_SHARDS = args.percent

edu_dir = 'edu_fineweb10B'
output_dir = f"data/fineweb-10pct-{NUM_SHARDS}"
os.makedirs(output_dir, exist_ok=True)

# ---- TRAINING SHARDS ----
# Get all train shards
train_shards = sorted([f for f in os.listdir(edu_dir) if f.endswith('.npy') and 'train' in f])
print(f"Found {len(train_shards)} train shards")

# Randomly select shards (out of 99 training shards)
num_to_select = NUM_SHARDS
selected_indices = sorted(random.sample(range(len(train_shards)), num_to_select))
selected_shards = [train_shards[i] for i in selected_indices]

print(f"Randomly selected {len(selected_shards)} shards: {selected_indices}")

train_output_path = os.path.join(output_dir, 'train.bin')

# Stream, not concat
total_tokens = 0
with open(train_output_path, "wb") as f:
    for shard_file in selected_shards:
        filepath = os.path.join(edu_dir, shard_file)

        shard = np.load(filepath)
        print(f"Loaded {shard_file}: {len(shard)} tokens")

        shard = shard.astype(np.uint16)
        shard.tofile(f) # write directly to disk

        total_tokens += len(shard)

print(f"\nTotal tokens: {total_tokens}")
print(f"Saved to {train_output_path}")

# ---- VALIDATION SHARD ----
val_file = 'edufineweb_val_000000.npy'
val_path = os.path.join(edu_dir, val_file)

val_tokens = np.load(val_path)
print(f"Loaded validation shard: {len(val_tokens)} tokens")

val_output_path = os.path.join(output_dir, 'val.bin')
val_tokens.astype(np.uint16).tofile(val_output_path)

print(f"Saved to {val_output_path}")