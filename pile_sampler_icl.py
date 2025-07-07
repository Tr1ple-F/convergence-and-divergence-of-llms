import ipdb
from datasets import load_dataset, VerificationMode
from transformers import AutoTokenizer
import numpy as np
import random

print("Loading dataset...")
ds = load_dataset("pietrolesci/pile-validation", "default", split="validation", verification_mode=VerificationMode.NO_CHECKS)
print("Dataset loaded.")

tokenizer = AutoTokenizer.from_pretrained(
    "EleutherAI/pythia-14m-seed1",
    revision="step128",
    cache_dir=f"./EleutherAI-pythia-14m-seed1/step128",
    add_bos_token=True,
    add_eos_token=True
)
ipdb.set_trace()
samples = []
for i in range(2000):
    samples.append(ds[random.randint(0, len(ds))])
ipdb.set_trace()

token_counter = 0

with open("working_dir/icl_score/input_text.txt", "w", encoding="utf-8") as file:
    for sample in samples:
        text = sample["text"]
        tokenized = tokenizer.encode(text)
        if sample["meta"] != "StackExchange" and sample["meta"] != 'GitHub' and len(tokenized) >= 505:
            file.write(text + "<|endoftext|>")
            token_counter += len(tokenized)

print(token_counter)