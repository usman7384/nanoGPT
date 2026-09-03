# Baseline nanoGPT architecture: LayerNorm + GELU + learned abs pos + AdamW + cosine.
# Sized so the non-embedding parameter count matches config/bench_modded.py exactly.

arch = 'baseline'
run_name = 'baseline'
dataset = 'finewebedu'

n_layer = 8
n_head = 8
n_embd = 512
block_size = 1024
bias = False          # nanoGPT's own recommended default
dropout = 0.0

total_tokens = int(300e6)
eval_every_tokens = int(10e6)

# bs=4 x accum=8 = 32768 tokens/iter. bs=16 peaks at 13.1GB > 12.88GB physical, and
# Windows silently spills to host memory instead of OOM-ing -- a 4.3x throughput cliff.
batch_size = 4
grad_accum = 8

learning_rate = 6e-4
weight_decay = 0.1
grad_clip = 1.0
schedule = 'cosine'
warmup_frac = 0.02
min_lr_frac = 0.1

compile = True
