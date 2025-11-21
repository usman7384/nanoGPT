import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset

dir_path = os.path.dirname(__file__)
train_txt = os.path.join(dir_path, 'train.txt')
val_txt = os.path.join(dir_path, 'val.txt')

# download tiny stories
dataset_train = load_dataset("roneneldan/TinyStories", split="train")
dataset_val = load_dataset("roneneldan/TinyStories", split="validation")

print("Writing train.txt...")
with open(train_txt, "w", encoding="utf-8") as f:
    for ex in tqdm(dataset_train, desc="Train split"):
        f.write(ex["text"].strip().replace("\n", " ") + "\n")

print("Writing val.txt...")
with open(val_txt, "w", encoding="utf-8") as f:
    for ex in tqdm(dataset_val, desc="Validation split"):
        f.write(ex["text"].strip().replace("\n", " ") + "\n")

# encode with gpt2 bpe
enc = tiktoken.get_encoding("gpt2")

def encode_file(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        data = f.read()
    ids = enc.encode_ordinary(data)
    return np.array(ids, dtype=np.uint16)

print("Encoding train...")
train_ids = encode_file(train_txt)
print("Encoding val...")
val_ids = encode_file(val_txt)

train_ids.tofile(os.path.join(dir_path, "train.bin"))
val_ids.tofile(os.path.join(dir_path, "val.bin"))

print("\nTinyStories prepared successfully!")
print(f"Train tokens: {len(train_ids):,}")
print(f"Val tokens:   {len(val_ids):,}")
