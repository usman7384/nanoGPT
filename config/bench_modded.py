# modded-nanogpt architecture: RoPE + RMSNorm + ReLU^2 + QK-norm + value embeddings
# + U-net skips + bigram hash + softcap, trained with Muon (hidden) + AdamW (embeds).
#
# Same n_layer/n_head/n_embd as the baseline, so NON-EMBEDDING params match. Total
# params are higher because modded-nanogpt unties embeddings and adds value/bigram
# tables -- set tie_embeddings=True and disable those for a strict iso-param run.

arch = 'modded'
run_name = 'modded'
dataset = 'finewebedu'

n_layer = 8
n_head = 8
n_embd = 512
block_size = 1024

total_tokens = int(300e6)
eval_every_tokens = int(10e6)

# identical batching to the baseline so batch composition is not a confound
batch_size = 4
grad_accum = 8

# architecture flags (flip any one to False to ablate it)
activation = 'relu2'
qk_norm = True
rope = True
zero_init_proj = True
tie_embeddings = False
logit_softcap = 30.0
value_embeddings = True
unet_skips = True
embed_skip = True
bigram_hash = True

# Muon for the transformer's 2D hidden weights, AdamW for embeddings/head/scalars
muon_lr = 0.02
learning_rate = 6e-4
weight_decay = 0.1
grad_clip = 0.0        # Muon's orthogonalized update is already norm-controlled
schedule = 'wsd'       # warmup -> stable -> linear cooldown
warmup_frac = 0.02
cooldown_frac = 0.4
min_lr_frac = 0.0

compile = True
