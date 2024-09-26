import random
from datasets import load_dataset

print("Loading dataset...")
ds = load_dataset("pietrolesci/pile-validation")
print("Dataset loaded.")
num_samples = 10

samples = random.sample(list(ds['validation']), num_samples)

with open("input_text.txt", "w", encoding="utf-8") as file:
    for sample in samples:
        file.write(sample['text'] + "<|endoftext|>")
