import os
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

ds = load_dataset("roneneldan/TinyStories")
print(ds)

tokenizer = AutoTokenizer.from_pretrained("gpt2")

def tokenize_and_save(dataset_split, tokenizer, out_path, batch_size=1000):
    texts = dataset_split["text"]
    total_tokens = 0

    with open(out_path, "wb") as f:
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            encoded = tokenizer(batch, add_special_tokens=False)["input_ids"]

            batch_ids = []
            for enc in encoded:
                ids = enc
                ids.append(tokenizer.eos_token_id)
                batch_ids.extend(ids)

            arr = np.array(batch_ids, dtype=np.uint16)
            arr.tofile(f)  # append this batch straight to disk
            total_tokens += len(arr)

            if i % 50000 == 0:
                print(f"Processed {i}/{len(texts)} stories")

    print(f"Saved {total_tokens} tokens to {out_path}")


tokenize_and_save(ds["train"], tokenizer, "train.bin")
tokenize_and_save(ds["validation"], tokenizer, "val.bin")