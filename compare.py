"""
Compare benchmark runs produced by train_bench.py.

Reports the two comparisons that matter and can disagree:
  - val loss at EQUAL TOKENS   (is the architecture more sample-efficient?)
  - val loss at EQUAL WALL-CLOCK (does it actually win on this GPU?)

A change that improves loss-per-token but costs 25% throughput can lose on wall-clock.
Both are reported so that tradeoff is visible rather than hidden.

Usage:
  python compare.py                          # every run in bench/
  python compare.py baseline modded          # specific runs
  python compare.py --plot                   # also write bench/comparison.png
"""

import os
import sys
import json
import glob

BENCH_DIR = 'bench'


def load(run_name):
    log = os.path.join(BENCH_DIR, f'{run_name}.jsonl')
    meta_path = os.path.join(BENCH_DIR, f'{run_name}.meta.json')
    if not os.path.exists(log):
        return None
    records = [json.loads(l) for l in open(log) if l.strip()]
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    return {'name': run_name, 'records': records, 'meta': meta}


def interp_at(records, key, target):
    """Linearly interpolate val_loss at a given tokens/time value."""
    pts = [(r[key], r['val_loss']) for r in records if r.get(key) is not None]
    pts.sort()
    if not pts or target < pts[0][0]:
        return None
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= target <= x1:
            if x1 == x0:
                return y1
            return y0 + (y1 - y0) * (target - x0) / (x1 - x0)
    return pts[-1][1] if target >= pts[-1][0] else None


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    do_plot = '--plot' in sys.argv

    if args:
        runs = [load(a) for a in args]
    else:
        names = sorted(os.path.basename(p)[:-6] for p in glob.glob(os.path.join(BENCH_DIR, '*.jsonl')))
        runs = [load(n) for n in names]
    runs = [r for r in runs if r and r['records']]
    if not runs:
        print(f"no runs found in {BENCH_DIR}/")
        return

    # common budgets: the smallest final tokens / training time across all runs
    common_tokens = min(r['records'][-1]['tokens'] for r in runs)
    common_time = min(r['records'][-1]['train_time_s'] for r in runs)

    print(f"\n{'='*104}")
    print(f"iso-token budget: {common_tokens/1e6:.0f}M tokens | iso-time budget: {common_time/60:.1f} min")
    print(f"{'='*104}")
    hdr = f"{'run':<22} {'params':>9} {'nonemb':>9} {'final':>8} {'@tokens':>9} {'@time':>9} {'tok/s':>9} {'MFU':>7} {'VRAM':>7}"
    print(hdr)
    print('-' * 104)

    rows = []
    for r in runs:
        last = r['records'][-1]
        meta = r['meta']
        v_tok = interp_at(r['records'], 'tokens', common_tokens)
        v_time = interp_at(r['records'], 'train_time_s', common_time)
        tps = max((x.get('tok_per_sec') or 0) for x in r['records'])
        mfu = max((x.get('mfu') or 0) for x in r['records'])
        rows.append((r['name'], v_tok, v_time))
        print(f"{r['name']:<22} "
              f"{meta.get('params_total',0)/1e6:>8.1f}M "
              f"{meta.get('params_nonembed',0)/1e6:>8.1f}M "
              f"{last['val_loss']:>8.4f} "
              f"{v_tok if v_tok is None else f'{v_tok:.4f}':>9} "
              f"{v_time if v_time is None else f'{v_time:.4f}':>9} "
              f"{tps/1e3:>8.0f}k "
              f"{mfu*100:>6.1f}% "
              f"{last['peak_vram_gb']:>6.2f}G")

    # deltas against the first run
    if len(rows) > 1:
        base_name, base_tok, base_time = rows[0]
        print(f"\ndelta vs '{base_name}' (negative = better):")
        for name, v_tok, v_time in rows[1:]:
            d_tok = f"{v_tok - base_tok:+.4f}" if (v_tok and base_tok) else 'n/a'
            d_time = f"{v_time - base_time:+.4f}" if (v_time and base_time) else 'n/a'
            print(f"  {name:<22} iso-token {d_tok:>9}   iso-time {d_time:>9}")
        print("\nnote: interpret deltas against your measured seed noise floor.")
        print("      run the same config twice with different --seed to establish it.")

    if do_plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print("\n--plot needs matplotlib: pip install matplotlib")
            return
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for r in runs:
            recs = [x for x in r['records'] if x['tokens'] > 0]
            axes[0].plot([x['tokens'] / 1e6 for x in recs], [x['val_loss'] for x in recs], label=r['name'], lw=1.8)
            axes[1].plot([x['train_time_s'] / 60 for x in recs], [x['val_loss'] for x in recs], label=r['name'], lw=1.8)
        axes[0].set_xlabel('tokens (M)'); axes[0].set_title('val loss vs tokens (sample efficiency)')
        axes[1].set_xlabel('training wall-clock (min)'); axes[1].set_title('val loss vs wall-clock (what you actually wait)')
        for ax in axes:
            ax.set_ylabel('val loss'); ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout()
        out = os.path.join(BENCH_DIR, 'comparison.png')
        fig.savefig(out, dpi=140)
        print(f"\nwrote {out}")


if __name__ == '__main__':
    main()
