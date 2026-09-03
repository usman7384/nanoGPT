"""
Muon: MomentUm Orthogonalized by Newton-schulz.

Single-GPU port of @KellerJordan's optimizer from modded-nanogpt. The original
shards the Newton-Schulz work across ranks; we have one GPU so every parameter is
orthogonalized locally.

References:
1) https://github.com/KellerJordan/Muon
2) https://github.com/KellerJordan/modded-nanogpt
3) https://kellerjordan.github.io/posts/muon/

Muon is only appropriate for 2D hidden-layer weights (the matmul parameters inside
the transformer blocks). Embeddings, the LM head, and anything 0/1-dimensional
(gains, biases, learnable scalars) should stay on AdamW -- see get_param_groups().
"""

import torch


def zeropower_via_newtonschulz5(G, steps=5):
    """
    Compute an approximate orthogonalization of G via a quintic Newton-Schulz iteration.

    We use coefficients chosen to maximise the slope of the iteration at zero, which
    makes convergence fast at the cost of not converging to exactly UV^T -- the
    singular values end up spread over roughly [0.7, 1.3] rather than all being 1.
    Empirically this does not hurt the optimizer, so we take the cheaper iteration.

    Runs in bfloat16: the iteration is numerically forgiving and this is ~2x faster.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    # the iteration converges for spectral norm <= 1, so normalise first.
    # transposing when tall means we always iterate on the smaller Gram matrix.
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    """
    Muon: SGD-momentum where the update is orthogonalized before being applied.

    The update for each 2D parameter is the Newton-Schulz orthogonalization of the
    (optionally Nesterov) momentum buffer. This makes the update's singular values
    roughly uniform, so no single direction dominates the step.

    Arguments:
        lr: learning rate. Much larger than AdamW's -- 0.02 is the modded-nanogpt
            default, because the orthogonalized update has unit-ish scale.
        momentum: momentum coefficient for the buffer.
        nesterov: use Nesterov-style momentum inside the update (recommended).
        ns_steps: Newton-Schulz iteration count. 5 is plenty.
        weight_decay: decoupled weight decay, applied directly to the parameter.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group['params']:
                assert p.ndim == 2, \
                    f"Muon only handles 2D parameters, got {p.ndim}D of shape {tuple(p.shape)}. " \
                    "Put embeddings/heads/gains/biases on AdamW instead."

    @torch.no_grad()
    def step(self, closure=None):
        """
        Parameters are bucketed by shape and orthogonalized in a single batched
        Newton-Schulz call per bucket.

        The naive one-tensor-at-a-time loop is launch-latency bound, not FLOP bound:
        49 tensors x 5 iterations x ~3 matmuls is ~735 tiny kernel launches per step,
        which measured 219ms -- more than the entire forward+backward pass. A GPT's
        hidden weights only occupy a handful of distinct shapes, so bucketing collapses
        that to one batched call each. zeropower_via_newtonschulz5 is already batched
        (it operates on the trailing two dims), so it needs no changes.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, momentum = group['lr'], group['momentum']

            # bucket by shape once, then reuse (param identity is stable across steps)
            if 'shape_buckets' not in group:
                buckets = {}
                for p in group['params']:
                    buckets.setdefault(tuple(p.shape), []).append(p)
                group['shape_buckets'] = list(buckets.values())

            for bucket in group['shape_buckets']:
                grads = [p.grad for p in bucket]
                if any(g is None for g in grads):
                    continue
                G = torch.stack(grads)                      # (k, m, n)

                state = self.state[bucket[0]]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(G)
                buf = state['momentum_buffer']
                buf.lerp_(G, 1 - momentum)
                # Nesterov: look ahead by blending the raw gradient back in
                update = G.lerp_(buf, momentum) if group['nesterov'] else buf

                O = zeropower_via_newtonschulz5(update, steps=group['ns_steps'])

                # every tensor in a bucket shares a shape, so one scale covers them all
                m, n = bucket[0].size(-2), bucket[0].size(-1)
                scale = max(1.0, m / n) ** 0.5
                if group['weight_decay'] != 0.0:
                    torch._foreach_mul_(bucket, 1 - lr * group['weight_decay'])
                # NS returns bfloat16; cast back so _foreach_add_ dtypes match the params
                torch._foreach_add_(bucket, list(O.type_as(bucket[0]).unbind(0)), alpha=-lr * scale)

        return loss


def get_param_groups(model):
    """
    Split a model's parameters into the Muon bank and the AdamW bank.

    Muon gets 2D weights from inside the transformer blocks. Everything else --
    token/value/bigram embeddings, the LM head, and all 0/1-dimensional tensors
    (RMSNorm gains if any, learnable skip lambdas, softcap scalars) -- goes to AdamW.
    Embeddings and the head are excluded deliberately: their gradients are sparse and
    row-structured, and orthogonalizing them empirically hurts.
    """
    muon_params, adamw_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_embed_or_head = ('embed' in name) or ('lm_head' in name) or ('wte' in name)
        if p.ndim == 2 and not is_embed_or_head:
            muon_params.append(p)
        else:
            adamw_params.append(p)
    return muon_params, adamw_params
