"""
Single-GPU training script for nanoGPT.

To run:
    python train.py --batch_size=32 --compile=False
    python train.py --dataset=openwebtext --max_iters=50000
    python train.py --init_from=resume  # resume from checkpoint

To override any hyperparameter from command line:
    python train.py --learning_rate=1e-3 --block_size=512
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from model import GPTConfig, GPT

# Add matplotlib for plotting losses (headless-friendly)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import textwrap

# ============================================================================
# Default Configuration
# ============================================================================

out_dir = '.'
eval_interval = 500
log_interval = 10
eval_iters = 100
eval_only = False
always_save_checkpoint = False
init_from = 'scratch'

wandb_log = False
wandb_project = 'nanogpt'
wandb_run_name = 'baseline'

dataset = 'openwebtext'
batch_size = 8
block_size = 1024

n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = True

learning_rate = 6e-4
max_iters = 50000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

decay_lr = True
warmup_iters = 1000
lr_decay_iters = max_iters
min_lr = 1e-5

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False
gradient_accumulation_steps = 1
optim = 'adamw' # 'adamw' or 'muon'

# ============================================================================
# Command-line overrides
# ============================================================================

config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())
config = {k: globals()[k] for k in config_keys}

# ============================================================================
# Setup
# ============================================================================

os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# ============================================================================
# Data Loading
# ============================================================================

data_dir = os.path.join('data', dataset)

def get_batch(split):
    """Load a batch of data from memory-mapped binary files."""
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

    if device_type == 'cuda':
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)

    return x, y

# ============================================================================
# Model Initialization
# ============================================================================

iter_num = 0
best_val_loss = 1e9

meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None

if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=None,
    dropout=dropout,
)

if init_from == 'scratch':
    print("Initializing a new model from scratch")
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2: 50257")

    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50257
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'checkpoint.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['config']

    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size', 'dropout']:
        if k in checkpoint_model_args:
            model_args[k] = checkpoint_model_args[k]

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'

    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

    if 'optim' in checkpoint:
        optim = checkpoint['optim']

model.to(device)

# ============================================================================
# Optimizer & Training Setup
# ============================================================================

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

opt_created = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type, optimizer_name=optim)
if isinstance(opt_created, tuple):
    # unpack muon + adam
    optimizers = list(opt_created)
else:
    optimizers = [opt_created]

if init_from == 'resume':
    # checkpoint may have saved optimizer state under 'optimizer' (single) or 'optimizers' (list)
    if 'optimizer' in checkpoint and len(optimizers) == 1:
        optimizers[0].load_state_dict(checkpoint['optimizer'])
    elif 'optimizers' in checkpoint:
        saved_list = checkpoint['optimizers']
        # naive mapping: load each into corresponding optimizer if lengths match
        if len(saved_list) == len(optimizers):
            for o, s in zip(optimizers, saved_list):
                o.load_state_dict(s)
        else:
            print("Warning: number of saved optimizers doesn't match current optimizers; skipping optimizer state restore")

checkpoint = None

if compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model)

# ============================================================================
# Loss estimation function
# ============================================================================

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):
            X, Y = get_batch(split)

            with ctx:
                logits, loss = model(X, Y)

            losses[k] = loss.item()

        out[split] = losses.mean()

    model.train()
    return out

# ============================================================================
# Plotting helper: save final training loss plot as PNG
# ============================================================================

def save_loss_plot(train_losses, config, out_dir='.', filename='loss_final.png'):
    """
    Save a PNG showing the training loss curve and a text box listing
    important run config values.

    Parameters:
      - train_losses: list of per-logging-iteration training loss floats
      - config: dict of config values
      - out_dir: directory to save png into
      - filename: filename for the png
    """
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    if len(train_losses) > 0:
        ax.plot(
            np.arange(1, len(train_losses) + 1),
            train_losses,
            label='train loss'
        )

    ax.set_xlabel('Logging step')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.grid(True, linestyle=':', linewidth=0.5)

    if len(train_losses) > 0:
        ax.legend()

    show_keys = [
        'n_layer',
        'n_head',
        'n_embd',
        'optim',
        'dropout',
        'batch_size',
        'block_size',
        'learning_rate',
        'max_iters',
        'weight_decay',
        'decay_lr',
        'warmup_iters',
    ]

    lines = []
    for k in show_keys:
        if k in config:
            lines.append(f"{k}: {config[k]}")

    extra = []
    for k in ('dataset', 'init_from', 'wandb_run_name', 'device', 'dtype'):
        if k in config:
            extra.append(f"{k}: {config[k]}")

    cfg_text = "\n".join(lines + [""] + extra)

    plt.tight_layout(rect=[0, 0, 0.72, 1.0])
    plt.gcf().text(
        0.74,
        0.5,
        textwrap.fill(cfg_text, width=40),
        fontsize=9,
        ha='left',
        va='center',
        family='monospace',
    )

    out_path = os.path.join(out_dir, filename)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved loss plot to {out_path}")

# ============================================================================
# Learning Rate Schedule
# ============================================================================

def get_lr(it):
    """Cosine learning rate decay with linear warmup."""
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)

    if it > lr_decay_iters:
        return min_lr

    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1

    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

def get_lr_multiplier(it):
    if it < warmup_iters:
        return (it + 1) / (warmup_iters + 1)

    progress = min(1.0, (it - warmup_iters) / max(1, lr_decay_iters - warmup_iters))
    coef = 0.5 * (1.0 + math.cos(math.pi * progress))
    return coef

# ============================================================================
# WandB logging
# ============================================================================

if wandb_log:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
print(f"tokens per iteration: {tokens_per_iter:,}")

# ============================================================================
# Training Loop
# ============================================================================

train_losses = []

X, Y = get_batch('train')
t0 = time.time()
running_mfu = -1.0

print(f"\nStarting training for {max_iters} iterations...")
print(f"Device: {device}, dtype: {dtype}\n")

while True:

    if len(optimizers) == 1: # AdamW
        lr = get_lr(iter_num) if decay_lr else learning_rate
    
        # set learning rate for all optimizers and their param groups
        for opt in optimizers:
            for param_group in opt.param_groups:
                param_group['lr'] = lr
    elif len(optimizers) == 2: # Muon and AdamW
        lr_multiplier = get_lr_multiplier(iter_num) if decay_lr else 1.0
        # scale each param_group from its stored base initial_lr (fallback to current lr)
        for opt in optimizers:
            for pg in opt.param_groups:
                base = pg.get('initial_lr', pg['lr'])
                pg['lr'] = min_lr + base * lr_multiplier * (1.0 - min_lr / base)

    # Evaluate and checkpoint
    if iter_num % eval_interval == 0:
        losses = estimate_loss()

        if len(optimizers) == 2:
            muon_lr = optimizers[0].param_groups[0]['lr']
            adamw_lr = optimizers[1].param_groups[0]['lr']
            print(
                f"step {iter_num:5d} | "
                f"train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f} | "
                f"muon_lr {muon_lr:.2e} | adamw_lr {adamw_lr:.2e}"
            )
        elif len(optimizers) == 1:
            adamw_lr = optimizers[0].param_groups[0]['lr']
            print(
                f"step {iter_num:5d} | "
                f"train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f} | "
                f"adamw_lr {adamw_lr:.2e}"
            )

        if wandb_log:
            if len(optimizers) == 2:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "muon_lr": muon_lr,
                    "adamw_lr": adamw_lr,
                    "mfu": running_mfu * 100,
                })
            else:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "adamw_lr": adamw_lr,
                    "mfu": running_mfu * 100,
                })

        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']

            if iter_num > 0:
                raw_model = model._orig_mod if compile else model
                
                checkpoint_dict = {
                    'model': raw_model.state_dict(),
                    'config': model_args,
                    'optim': optim,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                }
                
                # add optimizer states
                if len(optimizers) == 1:
                    checkpoint_dict['optimizer'] = optimizers[0].state_dict()
                else:
                    checkpoint_dict['optimizers'] = [o.state_dict() for o in optimizers]

                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint_dict, os.path.join(out_dir, 'checkpoint.pt'))

    if iter_num == 0 and eval_only:
        break

    # Forward-backward pass with gradient accumulation
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps

        X, Y = get_batch('train')
        scaler.scale(loss).backward()

    # Gradient clipping
    if grad_clip != 0.0:
        for opt in optimizers:
            try:
                scaler.unscale_(opt)
            except Exception:
                # custom optimizers like Muon may not be compatible with the GradScaler unscale API
                pass
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # Optimizer step
    for opt in optimizers:
        try:
            # scaler.step supports stepping only for optimizers that use CUDA amp scaler
            scaler.step(opt)
        except Exception:
            # if scaler.step fails for a non-supported optimizer (Muon likely doesn't use scaler), call opt.step()
            opt.step()
    
    scaler.update() # update scaler once
    
    # zero grads on model
    for opt in optimizers:
        try:
            opt.zero_grad(set_to_none=True)
        except TypeError:
            # Some custom optimizers may not accept set_to_none kw; fall back
            opt.zero_grad()

    # Timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    if iter_num % log_interval == 0:
        lossf = loss.item() * gradient_accumulation_steps
        train_losses.append(float(lossf))

        if iter_num > 5:
            raw_model = model._orig_mod if compile else model
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu

            print(
                f"  iter {iter_num:5d} | "
                f"loss {lossf:.4f} | "
                f"time {dt*1000:6.2f}ms | "
                f"mfu {running_mfu*100:5.2f}%"
            )
        else:
            print(
                f"  iter {iter_num:5d} | "
                f"loss {lossf:.4f} | "
                f"time {dt*1000:6.2f}ms"
            )

    iter_num += 1

    if iter_num > max_iters:
        break

# Save only the final plot at the end of training
try:
    save_loss_plot(train_losses, config, out_dir=out_dir, filename='loss_final.png')
except Exception as e:
    print(f"Warning: failed to save final loss plot: {e}")

print(f"\nTraining completed. Best val loss: {best_val_loss:.4f}")
print(f"Final checkpoint saved to {out_dir}/checkpoint.pt")