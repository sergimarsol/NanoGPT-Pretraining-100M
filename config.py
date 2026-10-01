import argparse
import json
import os
import torch

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint_dir", help="Directory containing checkpoint.pt")
args = parser.parse_args()

ckpt_path = os.path.join(args.checkpoint_dir, "checkpoint.pt")
config_path = os.path.join(args.checkpoint_dir, "config.json")

ckpt = torch.load(ckpt_path, map_location="cpu")

with open(config_path, "w") as f:
    json.dump(ckpt["config"], f, indent=2)

print(f"Saved config to {config_path}")