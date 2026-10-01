"""
CSE 251B — nanoGPT model.py

- SwiGLU MLP
- RoPE positional encoding (no learned wpe table)
- RMSNorm instead of LayerNorm
- QK-norm in attention
- WSD lr schedule

- Muon for all 2D params and AdamW for 1D params
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer


# ============================================================================
# Muon implementation
# Lightweight single-GPU orthogonalization-based Muon optimizer.
# ============================================================================

def zeropower_via_svd(G):
    # safe, but slow
    U, S, Vt = torch.linalg.svd(G, full_matrices=False)
    return U @ Vt


def zeropower_via_newtonschulz5(G, steps: int = 5, eps: float = 1e-7):
    # Newton-Schulz style orthogonalization (approximate UV^T)
    # Expects a 2D tensor input and returns a same-shape tensor.
    a = 3.4445
    b = -4.7750
    c = 2.0315
    X = G.to(torch.bfloat16)
    normval = X.norm() + eps
    X = X / normval
    transposed = False
    if X.size(0) > X.size(1):
        X = X.t()
        transposed = True
    for _ in range(steps):
        A = X @ X.t()
        B = b * A + c * (A @ A)
        X = a * X + (B @ X)
    if transposed:
        X = X.t()
    return X.to(G.dtype)


zeropower_backends = {
    "svd": zeropower_via_svd,
    "newtonschulz5": zeropower_via_newtonschulz5,
}


class Muon(Optimizer):
    """
    Muon: single-GPU orthogonalized update optimizer.
    Expects hidden 2D weight matrices. Use AdamW for embeddings, norms, biases, etc.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 backend="newtonschulz5", backend_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        backend=backend, backend_steps=backend_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            backend = group["backend"]
            backend_steps = group["backend_steps"]
            zeropower = zeropower_backends.get(backend, zeropower_via_newtonschulz5)

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                state = self.state[p]

                buf = state.get("momentum_buffer")
                if buf is None:
                    buf = state["momentum_buffer"] = torch.zeros_like(g)

                buf.mul_(momentum).add_(g)
                update = g + momentum * buf if nesterov else buf

                if p.ndim == 2:
                    try:
                        # New PyTorch warns about torch.cuda.amp.autocast, but this keeps
                        # compatibility with the rest of your current codebase.
                        with torch.cuda.amp.autocast(enabled=False):
                            ortho = zeropower(update, steps=backend_steps)
                    except Exception:
                        ortho = zeropower_via_svd(update)

                    # Scale heuristic from common Muon implementations.
                    scale = max(1.0, p.size(0) / max(1.0, p.size(1))) ** 0.5
                    p.add_(ortho * scale, alpha=-lr)
                else:
                    p.add_(update, alpha=-lr)
        return loss


# ============================================================================
# Model Architecture Components
# ============================================================================

class RMSNorm(nn.Module):
    """RMSNorm with learned scale."""

    def __init__(self, ndim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


def rotate_half(x):
    """Helper for RoPE. x shape: (..., head_dim), head_dim must be even."""
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


def apply_rope(q, k, cos, sin):
    """
    Apply rotary embeddings to q and k.
    q, k: (B, n_head, T, head_dim)
    cos, sin: (1, 1, T, head_dim)
    """
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with RoPE positional encoding."""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        assert self.head_dim % 2 == 0, "RoPE requires an even head_dim"

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        # RoPE frequencies. Stored as a buffer so it moves with the model/device.
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("rope_inv_freq", inv_freq, persistent=False)

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size))
                .view(1, 1, config.block_size, config.block_size),
            )

    def _rope_cos_sin(self, T, device, dtype):
        t = torch.arange(T, device=device, dtype=self.rope_inv_freq.dtype)
        freqs = torch.outer(t, self.rope_inv_freq)  # (T, head_dim/2)
        emb = torch.repeat_interleave(freqs, repeats=2, dim=-1)  # (T, head_dim)
        cos = emb.cos().to(dtype=dtype).view(1, 1, T, self.head_dim)
        sin = emb.sin().to(dtype=dtype).view(1, 1, T, self.head_dim)
        return cos, sin

    def forward(self, x):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE positional encoding on q/k instead of learned absolute position embeddings.
        cos, sin = self._rope_cos_sin(T, x.device, q.dtype)
        q, k = apply_rope(q, k, cos, sin)


        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Feed-forward network using SwiGLU instead of GELU."""

    def __init__(self, config):
        super().__init__()
        hidden_dim = int((8 / 3) * config.n_embd)
        hidden_dim = 8 * ((hidden_dim + 7) // 8)
        self.c_fc = nn.Linear(config.n_embd, 2 * hidden_dim, bias=config.bias)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x_gate, x_value = self.c_fc(x).chunk(2, dim=-1)
        x = F.silu(x_gate) * x_value
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer block: communication followed by computation."""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    """Configuration for GPT model."""
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False


class GPT(nn.Module):
    """GPT Language Model."""

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=RMSNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight tying
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """Return the number of parameters in the model."""
        # With RoPE there is no learned positional embedding table to subtract.
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        tok_emb = self.transformer.wte(idx)
        x = self.transformer.drop(tok_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

    def crop_block_size(self, block_size):
        """Model surgery to decrease the block size if necessary."""
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, optimizer_name='adamw'):
        """
        Configure optimizer for training.
        AdamW path: standard decay/no-decay split.
        Muon path: original model.py behavior: Muon for all 2D parameters and AdamW for 1D parameters.
        """
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]

        print(f"num decayed parameter tensors: {len(decay_params)}, with {sum(p.numel() for p in decay_params):,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {sum(p.numel() for p in nodecay_params):,} parameters")

        optimizer_name = optimizer_name.lower()
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()

        if optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
            print(f"using fused AdamW: {use_fused}")
            return optimizer

        elif optimizer_name == 'muon':
            muon_lr = 0.003
            adamw_lr = 1e-4

            muon_opt = Muon(
                decay_params,
                lr=muon_lr,
                momentum=0.95,
                nesterov=True,
                backend='newtonschulz5',
                backend_steps=5,
            )

            adam_for_1d = torch.optim.AdamW(
                nodecay_params,
                lr=adamw_lr,
                betas=betas,
                **extra_args,
            )

            for pg in muon_opt.param_groups:
                pg['initial_lr'] = muon_lr
            for pg in adam_for_1d.param_groups:
                pg['initial_lr'] = adamw_lr

            print(f"using Muon for all 2D params and AdamW for 1D params (fused: {use_fused})")
            return (muon_opt, adam_for_1d)

        else:
            raise ValueError(f"Unknown optimizer_name: {optimizer_name}")

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """Estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS."""
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12
        return flops_achieved / flops_promised

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate text from a conditioning sequence."""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ============================================================================
# Submission Interface
# ============================================================================

class GPTForEvaluation(nn.Module):
    """Wrapper for evaluation that ensures model returns only logits."""

    def __init__(self, gpt_model):
        super().__init__()
        self.gpt = gpt_model

    def forward(self, input_ids):
        logits, _ = self.gpt(input_ids, targets=None)
        return logits


def load_model(checkpoint_path: str, device: str = "cuda") -> torch.nn.Module:
    """Load your trained model from a checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = GPTConfig(**checkpoint['config'])
    model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    eval_model = GPTForEvaluation(model)
    eval_model.to(device)
    eval_model.eval()
    return eval_model


if __name__ == "__main__":
    print("Creating GPT model...")
    config = GPTConfig()
    model = GPT(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")

    dummy_input = torch.randint(0, 50257, (2, 1024))
    logits, loss = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {logits.shape}")
    assert logits.shape == (2, 1024, 50257), "Output shape mismatch!"
    print("Interface check passed.")
