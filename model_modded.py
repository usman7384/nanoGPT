"""
Modernized GPT following @KellerJordan's modded-nanogpt.

Every modification is behind a flag in ModdedGPTConfig so components can be ablated
one at a time rather than only compared as a bundle. Defaults are all-on.

Ported from modded-nanogpt:
  - RoPE instead of learned absolute position embeddings
  - parameter-free RMSNorm instead of LayerNorm
  - ReLU^2 MLP instead of GELU  (note: NOT SwiGLU -- @KellerJordan found ReLU^2 better)
  - QK-norm on the query/key projections
  - no biases anywhere
  - zero-init output projections and LM head (muP-like)
  - untied embeddings
  - softcapped logits
  - value embeddings mixed into attention V
  - U-net style skip connections across blocks
  - embedding skip connection into every block
  - bigram hash embedding on 1/4 of model_dim with a sign trick
  - Muon on hidden matrices, AdamW on embeddings/head/scalars (see muon.py)

Deliberately NOT ported, because they are 8xH100 systems work rather than
architecture and do not transfer to a single 12GB laptop GPU:
  - FP8 matmuls and the custom torch.ops.nanogpt kernels (Ada fp8 support is
    immature; these target Hopper)
  - FlexAttention long-short sliding windows + YaRN window warmup (the win shows up
    at 48k+ context; we train at 1024, and Windows torch has no flash backend)
  - distributed/sharded Muon (single GPU)
  - MUDD connections, hyperconnections, XSA, paired-head attention, multi-token
    prediction (later records, entangled with their exact 8xH100 config)
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


def norm(x):
    """Parameter-free RMSNorm, as used throughout modded-nanogpt."""
    return F.rms_norm(x, (x.size(-1),))


# -----------------------------------------------------------------------------
# rotary position embeddings

class Rotary(nn.Module):
    """
    The cos/sin tables are built once at init for the full block_size. They were
    originally built lazily inside forward via register_buffer, which mutates module
    state during the forward pass -- that forces a torch.compile graph break on every
    step and measured ~2x slower end-to-end. Precomputing keeps the graph static.
    """

    def __init__(self, head_dim, block_size, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        pos = torch.arange(block_size, dtype=torch.float32)
        freqs = torch.outer(pos, inv_freq)
        self.register_buffer('cos_cached', freqs.cos(), persistent=False)
        self.register_buffer('sin_cached', freqs.sin(), persistent=False)

    def forward(self, x):
        # x: (B, n_head, T, head_dim)
        t = x.size(-2)
        cos = self.cos_cached[:t]
        sin = self.sin_cached[:t]
        x1, x2 = x.float().chunk(2, dim=-1)
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        return torch.cat((y1, y2), dim=-1).type_as(x)


# -----------------------------------------------------------------------------
# bigram hash embedding
#
# NOTE: reconstructed from the modded-nanogpt README description ("Bigram hash
# embedding on 1/4 of model_dim w/ sign trick") rather than ported verbatim -- the
# fetch tooling could not return the source. The idea: hash each (prev, cur) token
# pair into a fixed-size table so the model gets an explicit bigram feature without
# a vocab^2 parameter cost. The sign trick (a second hash choosing +/-1) halves
# collision bias, since colliding pairs cancel in expectation instead of summing.

class BigramHashEmbedding(nn.Module):
    def __init__(self, n_embd, n_buckets=2**16, frac=4):
        super().__init__()
        self.dim = n_embd // frac
        self.n_buckets = n_buckets
        self.table = nn.Embedding(n_buckets, self.dim)
        nn.init.normal_(self.table.weight, std=0.02)

    def forward(self, idx):
        # pair each token with its predecessor (first position pairs with itself).
        # built with cat rather than roll+in-place assignment to stay compile-friendly.
        prev = torch.cat([idx[:, :1], idx[:, :-1]], dim=1)
        # cheap multiplicative hash over the pair
        h = (prev.long() * 0x9E3779B1) ^ (idx.long() * 0x85EBCA77)
        h = h & 0x7FFFFFFF
        bucket = h % self.n_buckets
        sign = torch.where((h >> 20 & 1).bool(), 1.0, -1.0).unsqueeze(-1)
        return self.table(bucket) * sign


# -----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):

    def __init__(self, config, layer_idx):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.layer_idx = layer_idx
        self.config = config

        self.c_q = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_k = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_v = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        if config.zero_init_proj:
            nn.init.zeros_(self.c_proj.weight)

        self.rotary = Rotary(self.head_dim, config.block_size, base=config.rope_base) if config.rope else None
        if config.pos_embd == 'learned':
            self.rotary = None

        # learnable mix between the projected V and the value embedding
        self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5])) if config.value_embeddings else None

    def forward(self, x, ve=None):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.config.qk_norm:
            q, k = norm(q), norm(k)
        if self.rotary is not None:
            q, k = self.rotary(q), self.rotary(k)

        # mix the value embedding into V for the layers that have one
        if ve is not None and self.lambdas is not None:
            ve = ve.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            v = self.lambdas[0] * v + self.lambdas[1] * ve

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        if config.zero_init_proj:
            nn.init.zeros_(self.c_proj.weight)
        self.activation = config.activation

    def forward(self, x):
        x = self.c_fc(x)
        if self.activation == 'relu2':
            x = F.relu(x).square()
        else:
            x = F.gelu(x)
        return self.c_proj(x)


class Block(nn.Module):

    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)
        # blend between the running residual and the original embedding
        self.lambdas = nn.Parameter(torch.tensor([1.0, 0.0])) if config.embed_skip else None

    def forward(self, x, x0=None, ve=None):
        if self.lambdas is not None and x0 is not None:
            x = self.lambdas[0] * x + self.lambdas[1] * x0
        x = x + self.attn(norm(x), ve=ve)
        x = x + self.mlp(norm(x))
        return x


@dataclass
class ModdedGPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    # --- ablation flags, all defaulting to the modded-nanogpt behaviour ---
    pos_embd: str = 'rope'          # 'rope' | 'learned'
    rope: bool = True
    rope_base: float = 10000.0
    activation: str = 'relu2'       # 'relu2' | 'gelu'
    qk_norm: bool = True
    zero_init_proj: bool = True
    tie_embeddings: bool = False    # modded-nanogpt unties; set True for iso-param
    logit_softcap: float = 30.0     # 0.0 disables
    value_embeddings: bool = True
    n_value_embeds: int = 3         # each is a full vocab x n_embd table -- see note in ModdedGPT
    unet_skips: bool = True
    embed_skip: bool = True
    bigram_hash: bool = True
    bigram_buckets: int = 2 ** 16


class ModdedGPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        nn.init.normal_(self.wte.weight, std=0.02)
        self.wpe = nn.Embedding(config.block_size, config.n_embd) if config.pos_embd == 'learned' else None

        self.bigram = BigramHashEmbedding(config.n_embd, config.bigram_buckets) if config.bigram_hash else None

        # Value embeddings: tables shared between early and late layers, mirroring the
        # U-net structure. NOTE these are expensive -- each is a full vocab x n_embd
        # table, so 3 of them at n_embd=512 is ~77M params, 3x the entire transformer.
        # That ratio is faithful to modded-nanogpt (their 3x768 tables are ~116M against
        # a 124M model), but it means total params are NOT comparable to the baseline.
        # Non-embedding params are what's matched. Lower n_value_embeds to shrink it.
        self.n_ve = min(config.n_value_embeds, config.n_layer // 2) if config.value_embeddings else 0
        if self.n_ve:
            self.value_embeds = nn.ModuleList([
                nn.Embedding(config.vocab_size, config.n_embd) for _ in range(self.n_ve)
            ])
            for e in self.value_embeds:
                nn.init.normal_(e.weight, std=0.02)

        self.h = nn.ModuleList([Block(config, i) for i in range(config.n_layer)])

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.zero_init_proj:
            nn.init.zeros_(self.lm_head.weight)
        else:
            nn.init.normal_(self.lm_head.weight, std=0.02)
        if config.tie_embeddings:
            self.wte.weight = self.lm_head.weight

        # U-net: first half encodes, second half decodes with skips from the first
        self.n_encoder = config.n_layer // 2
        if config.unet_skips:
            self.skip_weights = nn.Parameter(torch.ones(config.n_layer - self.n_encoder))

        print("number of parameters: %.2fM (non-embedding: %.2fM)"
              % (self.get_num_params() / 1e6, self.get_num_params(non_embedding=True) / 1e6))

    def get_num_params(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.wte.weight.numel()
            if self.wpe is not None:
                n -= self.wpe.weight.numel()
            if self.bigram is not None:
                n -= self.bigram.table.weight.numel()
            if self.n_ve:
                n -= sum(e.weight.numel() for e in self.value_embeds)
            if not self.config.tie_embeddings:
                n -= self.lm_head.weight.numel()
        return n

    def _layer_ve(self, i, ves):
        """Map a layer index to its value embedding, U-net style: first n and last n."""
        if not self.n_ve:
            return None
        if i < self.n_ve:
            return ves[i]
        if i >= self.config.n_layer - self.n_ve:
            return ves[self.config.n_layer - 1 - i]
        return None

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"sequence length {T} > block size {self.config.block_size}"

        x = self.wte(idx)
        if self.wpe is not None:
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
            x = x + self.wpe(pos)
        if self.bigram is not None:
            # add the bigram feature into the first quarter of the channels
            bg = self.bigram(idx)
            x = torch.cat([x[..., :bg.size(-1)] + bg, x[..., bg.size(-1):]], dim=-1)

        x = norm(x)
        x0 = x
        ves = [e(idx) for e in self.value_embeds] if self.n_ve else None

        skips = []
        for i, block in enumerate(self.h):
            if self.config.unet_skips and i >= self.n_encoder:
                x = x + self.skip_weights[i - self.n_encoder] * skips.pop()
            x = block(x, x0=x0, ve=self._layer_ve(i, ves))
            if self.config.unet_skips and i < self.n_encoder:
                skips.append(x)

        x = norm(x)

        if targets is not None:
            logits = self.lm_head(x)
            logits = self._softcap(logits)
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self._softcap(self.lm_head(x[:, [-1], :]))
            loss = None
        return logits, loss

    def _softcap(self, logits):
        c = self.config.logit_softcap
        if c and c > 0:
            return c * torch.tanh(logits / c)
        return logits

    def configure_optimizers(self, muon_lr, adamw_lr, weight_decay, betas, device_type):
        """Muon on the transformer's 2D hidden weights, AdamW on everything else."""
        from muon import Muon, get_param_groups
        muon_params, adamw_params = get_param_groups(self)
        n_muon = sum(p.numel() for p in muon_params)
        n_adamw = sum(p.numel() for p in adamw_params)
        print(f"Muon: {len(muon_params)} tensors, {n_muon:,} params")
        print(f"AdamW: {len(adamw_params)} tensors, {n_adamw:,} params")

        import inspect
        fused = 'fused' in inspect.signature(torch.optim.AdamW).parameters and device_type == 'cuda'
        optimizers = [
            Muon(muon_params, lr=muon_lr, momentum=0.95, nesterov=True, ns_steps=5),
            torch.optim.AdamW(adamw_params, lr=adamw_lr, betas=betas,
                              weight_decay=weight_decay, **(dict(fused=True) if fused else {})),
        ]
        return optimizers

    def estimate_mfu(self, fwdbwd_per_iter, dt, flops_promised=47.2e12):
        """
        MFU against this machine's *measured* bf16 matmul ceiling (47.2 TFLOPS on a
        105W-capped 4080 Laptop), not A100 peak, so the number is meaningful here.
        Uses the same 6N + 12*L*H*Q*T formula as the baseline for comparability.
        """
        N = self.get_num_params(non_embedding=True)
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_iter = flops_per_token * T * fwdbwd_per_iter
        return (flops_per_iter * (1.0 / dt)) / flops_promised

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
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
