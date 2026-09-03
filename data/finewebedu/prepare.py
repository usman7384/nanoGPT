"""
Prepare a fixed-size slice of FineWeb-Edu for the architecture benchmark.

We stream the dataset rather than downloading it, and stop the moment we have the
requested number of tokens. This matters here: the full sample-10BT is ~28GB and
this machine has ~20GB free. Streaming caps disk usage at exactly the output bins.

Output (default 300M tokens, GPT-2 BPE, uint16):
  train.bin  ~594MB
  val.bin    ~6MB

Usage:
  python data/finewebedu/prepare.py                      # 300M tokens
  python data/finewebedu/prepare.py --total_tokens=1e9   # bigger run
"""

import os
import sys
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# -----------------------------------------------------------------------------
total_tokens = int(300e6)   # total tokens to write across train+val
val_tokens = int(3e6)       # held-out validation tokens
dataset_name = "HuggingFaceFW/fineweb-edu"
dataset_config = "sample-10BT"
shard_size = int(1e7)       # tokens buffered in RAM before flushing to disk
# -----------------------------------------------------------------------------
for arg in sys.argv[1:]:
    assert arg.startswith('--') and '=' in arg, f"bad arg: {arg}"
    key, val = arg[2:].split('=')
    assert key in globals(), f"unknown key: {key}"
    globals()[key] = type(globals()[key])(float(val)) if isinstance(globals()[key], int) else type(globals()[key])(val)
    print(f"Overriding: {key} = {globals()[key]}")

os.makedirs(os.path.dirname(__file__), exist_ok=True)
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']  # document delimiter


def tokenize(doc):
    # prepend EOT so the model sees an explicit document boundary
    ids = [eot] + enc.encode_ordinary(doc["text"])
    arr = np.array(ids, dtype=np.uint16)
    assert (arr < 2**16).all(), "token id out of uint16 range"
    return arr


def write_split(stream, n_tokens, path, desc):
    """Pull from `stream` until n_tokens are written to `path`. Returns tokens written."""
    buf = []
    written = 0
    with open(path, 'wb') as f, tqdm(total=n_tokens, unit='tok', unit_scale=True, desc=desc) as pbar:
        for doc in stream:
            arr = tokenize(doc)
            buf.append(arr)
            written += len(arr)
            pbar.update(len(arr))
            if written >= n_tokens:
                break
            if sum(len(a) for a in buf) >= shard_size:
                np.concatenate(buf).tofile(f)
                buf = []
        if buf:
            np.concatenate(buf).tofile(f)
    return written


if __name__ == '__main__':
    train_tokens = total_tokens - val_tokens
    print(f"streaming {dataset_name} ({dataset_config})")
    print(f"target: {train_tokens/1e6:.0f}M train + {val_tokens/1e6:.0f}M val tokens")
    print(f"estimated disk: {2*total_tokens/1e9:.2f} GB")

    ds = load_dataset(dataset_name, name=dataset_config, split="train", streaming=True)
    stream = iter(ds)

    # val first, so train and val come from disjoint documents
    n_val = write_split(stream, val_tokens, os.path.join(os.path.dirname(__file__), 'val.bin'), 'val')
    n_train = write_split(stream, train_tokens, os.path.join(os.path.dirname(__file__), 'train.bin'), 'train')

    print(f"\nwrote val.bin:   {n_val:,} tokens")
    print(f"wrote train.bin: {n_train:,} tokens")
