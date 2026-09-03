"""
Benchmark harness for comparing the baseline nanoGPT architecture against the
modded-nanogpt architecture under identical conditions.

train.py is left untouched as the reference implementation; this script drives both
architectures through the same data loader, seed, and token budget, and logs
everything needed to compare them at equal tokens AND equal wall-clock.

Usage:
  python train_bench.py config/bench_baseline.py
  python train_bench.py config/bench_modded.py
  python train_bench.py config/bench_modded.py --muon_lr=0.01 --run_name=modded_lr01

Writes bench/<run_name>.jsonl (one record per eval) and bench/<run_name>.meta.json.
"""

import os
import json
import math
import time
from contextlib import nullcontext

import numpy as np
import torch

# -----------------------------------------------------------------------------
# defaults -- override via config file or --flag=value
arch = 'baseline'            # 'baseline' | 'modded'
run_name = 'run'
out_dir = 'bench'
dataset = 'finewebedu'

# token budget (the thing held constant across architectures)
total_tokens = int(300e6)
eval_every_tokens = int(10e6)
eval_batches = 40            # fixed val batches, same for every run

# model (kept identical between architectures)
n_layer = 8
n_head = 8
n_embd = 512
block_size = 1024
dropout = 0.0
bias = False                 # baseline only; modded is always bias-free

# modded ablation flags
activation = 'relu2'
qk_norm = True
rope = True
zero_init_proj = True
tie_embeddings = False
logit_softcap = 30.0
value_embeddings = True
n_value_embeds = 3
unet_skips = True
embed_skip = True
bigram_hash = True

# batching
batch_size = 16
grad_accum = 2

# optimizer
learning_rate = 6e-4         # AdamW lr (baseline: the only lr; modded: embeds/head/scalars)
muon_lr = 0.02               # modded only
weight_decay = 0.1
beta1, beta2 = 0.9, 0.95
grad_clip = 1.0              # 0 disables

# schedule
schedule = 'cosine'          # 'cosine' | 'wsd'
warmup_frac = 0.02
cooldown_frac = 0.4          # wsd only
min_lr_frac = 0.1

# wandb (same knobs as upstream train.py)
wandb_log = False
wandb_project = 'nanogpt-arch-bench'
wandb_run_name = ''          # defaults to run_name

# system
device = 'cuda'
dtype = 'bfloat16'
compile = True
seed = 1337
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

tokens_per_iter = batch_size * grad_accum * block_size
max_iters = total_tokens // tokens_per_iter
eval_every_iters = max(1, eval_every_tokens // tokens_per_iter)
print(f"tokens/iter: {tokens_per_iter:,} | iters: {max_iters:,} | eval every {eval_every_iters} iters")

# -----------------------------------------------------------------------------
# data

data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
print(f"train: {len(train_data):,} tokens | val: {len(val_data):,} tokens")

# fixed validation batches -- identical across every run so val losses are comparable
_val_rng = np.random.default_rng(12345)
val_offsets = [_val_rng.integers(0, len(val_data) - block_size - 1, size=batch_size) for _ in range(eval_batches)]

# training sampler seeded per-run but identical across architectures (same `seed`)
_train_rng = np.random.default_rng(seed)


def _make_batch(data, offsets):
    x = torch.from_numpy(np.stack([data[i:i + block_size].astype(np.int64) for i in offsets]))
    y = torch.from_numpy(np.stack([data[i + 1:i + 1 + block_size].astype(np.int64) for i in offsets]))
    if device_type == 'cuda':
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


def get_train_batch():
    offsets = _train_rng.integers(0, len(train_data) - block_size - 1, size=batch_size)
    return _make_batch(train_data, offsets)


def get_val_batch(k):
    return _make_batch(val_data, val_offsets[k])


# -----------------------------------------------------------------------------
# model

if arch == 'baseline':
    from model import GPT, GPTConfig
    model = GPT(GPTConfig(block_size=block_size, vocab_size=50304, n_layer=n_layer,
                          n_head=n_head, n_embd=n_embd, dropout=dropout, bias=bias))
elif arch == 'modded':
    from model_modded import ModdedGPT, ModdedGPTConfig
    model = ModdedGPT(ModdedGPTConfig(
        block_size=block_size, vocab_size=50304, n_layer=n_layer, n_head=n_head, n_embd=n_embd,
        activation=activation, qk_norm=qk_norm, rope=rope, zero_init_proj=zero_init_proj,
        tie_embeddings=tie_embeddings, logit_softcap=logit_softcap,
        value_embeddings=value_embeddings, n_value_embeds=n_value_embeds,
        unet_skips=unet_skips, embed_skip=embed_skip, bigram_hash=bigram_hash))
else:
    raise ValueError(f"unknown arch: {arch}")

model.to(device)
n_params_total = sum(p.numel() for p in model.parameters())
n_params_nonembed = model.get_num_params(non_embedding=True) if arch == 'modded' else model.get_num_params()

# optimizers (modded returns two: Muon + AdamW)
if arch == 'baseline':
    optimizers = [model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)]
    base_lrs = [learning_rate]
else:
    optimizers = model.configure_optimizers(muon_lr, learning_rate, weight_decay, (beta1, beta2), device_type)
    base_lrs = [muon_lr, learning_rate]

raw_model = model
if compile:
    print("compiling model...")
    model = torch.compile(model)


def get_lr_mult(it):
    """Returns a multiplier on each optimizer's base lr."""
    warmup = max(1, int(warmup_frac * max_iters))
    if it < warmup:
        return (it + 1) / (warmup + 1)
    if schedule == 'cosine':
        ratio = (it - warmup) / max(1, max_iters - warmup)
        ratio = min(1.0, max(0.0, ratio))
        return min_lr_frac + 0.5 * (1 + math.cos(math.pi * ratio)) * (1 - min_lr_frac)
    elif schedule == 'wsd':
        # warmup -> stable -> linear cooldown
        cooldown_start = int((1 - cooldown_frac) * max_iters)
        if it < cooldown_start:
            return 1.0
        ratio = (it - cooldown_start) / max(1, max_iters - cooldown_start)
        return 1.0 - (1 - min_lr_frac) * ratio
    raise ValueError(schedule)


# MFU is computed here rather than via model.estimate_mfu() so both architectures use
# identical accounting. model.py's version hardcodes A100 peak (312 TFLOPS); we measure
# against this machine's actual bf16 matmul ceiling.
PEAK_FLOPS = 47.2e12  # measured: 4096^3 bf16 matmul on this 105W-capped 4080 Laptop


def estimate_mfu(dt):
    L, H, Q, T = n_layer, n_head, n_embd // n_head, block_size
    flops_per_token = 6 * n_params_nonembed + 12 * L * H * Q * T
    flops_per_iter = flops_per_token * T * (batch_size * grad_accum)
    return (flops_per_iter / dt) / PEAK_FLOPS


@torch.no_grad()
def evaluate():
    model.eval()
    losses = torch.zeros(eval_batches)
    for k in range(eval_batches):
        X, Y = get_val_batch(k)
        with ctx:
            _, loss = model(X, Y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


# -----------------------------------------------------------------------------
# train

log_path = os.path.join(out_dir, f'{run_name}.jsonl')
meta_path = os.path.join(out_dir, f'{run_name}.meta.json')
with open(meta_path, 'w') as f:
    json.dump({'config': config, 'params_total': n_params_total,
               'params_nonembed': n_params_nonembed, 'max_iters': max_iters,
               'tokens_per_iter': tokens_per_iter}, f, indent=2)
logf = open(log_path, 'w')

if wandb_log:
    import wandb
    wandb.init(project=wandb_project, name=(wandb_run_name or run_name),
               config={**config, 'params_total': n_params_total,
                       'params_nonembed': n_params_nonembed})
    # plot against tokens rather than step, so runs with different throughput
    # remain directly comparable on the x-axis
    wandb.define_metric('tokens')
    wandb.define_metric('*', step_metric='tokens')

print(f"\n=== {run_name} | arch={arch} | {n_params_total/1e6:.1f}M params "
      f"({n_params_nonembed/1e6:.1f}M non-embed) ===")

torch.cuda.reset_peak_memory_stats()
X, Y = get_train_batch()
train_time = 0.0          # pure training seconds, excluding eval
t_stretch = time.time()   # start of the current uninterrupted training stretch
t_mfu, iters_since_mfu = time.time(), 0
running_mfu = -1.0
tok_per_sec = 0.0

for it in range(max_iters + 1):
    mult = get_lr_mult(it)
    for opt, base in zip(optimizers, base_lrs):
        for g in opt.param_groups:
            g['lr'] = base * mult

    # eval + log
    if it % eval_every_iters == 0 or it == max_iters:
        torch.cuda.synchronize()
        train_time += time.time() - t_stretch
        val_loss = evaluate()
        rec = {
            'iter': it,
            'tokens': it * tokens_per_iter,
            'train_time_s': round(train_time, 2),
            'val_loss': round(val_loss, 4),
            'lr_mult': round(mult, 4),
            'mfu': round(running_mfu, 4) if running_mfu > 0 else None,
            'tok_per_sec': round(tok_per_sec),
            'peak_vram_gb': round(torch.cuda.max_memory_allocated() / 1e9, 2),
        }
        logf.write(json.dumps(rec) + '\n')
        logf.flush()
        if wandb_log:
            wandb.log({k: v for k, v in rec.items() if v is not None})
        print(f"[{run_name}] iter {it}/{max_iters} | tokens {it*tokens_per_iter/1e6:.0f}M | "
              f"val {val_loss:.4f} | {train_time/60:.1f}min | mfu {running_mfu*100:.1f}% | "
              f"{tok_per_sec/1e3:.0f}k tok/s | vram {rec['peak_vram_gb']:.2f}GB")
        # restart both timers so eval time is excluded from training wall-clock
        t_stretch = time.time()
        t_mfu, iters_since_mfu = time.time(), 0

    if it == max_iters:
        break

    for micro in range(grad_accum):
        with ctx:
            _, loss = model(X, Y)
            loss = loss / grad_accum
        X, Y = get_train_batch()
        loss.backward()

    if grad_clip != 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    for opt in optimizers:
        opt.step()
    for opt in optimizers:
        opt.zero_grad(set_to_none=True)

    # throughput/MFU on its own timer so it never disturbs `train_time`
    iters_since_mfu += 1
    if iters_since_mfu >= 20:
        torch.cuda.synchronize()
        dt = (time.time() - t_mfu) / iters_since_mfu
        mfu = estimate_mfu(dt)
        running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        tok_per_sec = tokens_per_iter / dt
        t_mfu, iters_since_mfu = time.time(), 0

logf.close()
print(f"\ndone: {run_name} | final val {rec['val_loss']:.4f} | "
      f"{train_time/60:.1f} min | peak vram {rec['peak_vram_gb']:.2f}GB")
print(f"log: {log_path}")
