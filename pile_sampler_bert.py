import ipdb
from datasets import load_dataset, VerificationMode
from transformers import AutoTokenizer
import numpy as np
import random

print("Loading dataset...")
ds = load_dataset("pietrolesci/pile-validation", "default", split="validation", verification_mode=VerificationMode.NO_CHECKS)
print("Dataset loaded.")

model_name = "google/multiberts-seed_0"

tokenizer = AutoTokenizer.from_pretrained(
    model_name + "-step_0k",
    cache_dir=f"./run-stage/{model_name.replace('/', '-')}/step_0k",
    add_bos_token=True,
    add_eos_token=True
)
ipdb.set_trace()
samples = []
for i in range(100):
    samples.append(ds[random.randint(0, len(ds))])
ipdb.set_trace()

token_counter = 0

with open("working_dir/input_text.txt", "w", encoding="utf-8") as file:
    for sample in samples:
        text = sample["text"]
        tokenized = tokenizer.encode(text)
        if sample["meta"] != "StackExchange" and sample["meta"] != 'GitHub' and len(tokenized) <= 511:
            file.write(text + "<|endoftext|>")
            token_counter += len(tokenized)

print(token_counter)