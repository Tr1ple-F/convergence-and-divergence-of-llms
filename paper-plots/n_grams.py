from datasets import load_dataset, VerificationMode
import random
import os
import numpy as np
from collections import defaultdict, Counter
import pickle

# Class to represent n-grams
class NGramModel:
    def __init__(self, n, num_batches, load_cache = True, train_cached=False):
        self.n = n
        self.num_batches = num_batches
        self.ngrams = defaultdict(Counter)
        self.cache_dir = './n_gram_models'
        self.unchanged = False

        os.makedirs(self.cache_dir, exist_ok=True)

        self.cache_path = os.path.join(self.cache_dir, f'{n}gram_model_{num_batches}b.pkl')

        if load_cache and os.path.exists(self.cache_path):
            self.load()
            self.unchanged = not train_cached

    def update(self, tokens):
        if self.unchanged:
            return

        if len(tokens) < self.n:
            return

        for i in range(len(tokens) - self.n + 1):
            context = tuple(tokens[i:i+self.n-1])
            next_word = tokens[i+self.n-1]
            self.ngrams[context][next_word] += 1

    def infer(self, tokens):
        context = tuple(tokens[-(self.n-1):])  # Take last n-1 tokens
        counter = self.ngrams.get(context, Counter())

        total = sum(counter.values())
        if total == 0:
            return {}

        distribution = {word: count / total for word, count in counter.items()}
        return distribution

    def cache(self):
        if self.unchanged:
            return
        with open(self.cache_path, 'wb') as f:
            pickle.dump(dict(self.ngrams), f)

    def load(self):
        with open(self.cache_path, 'rb') as f:
            data = pickle.load(f)
            self.ngrams = defaultdict(Counter, data)


# Function to get $n$ batches worth of tokens
def get_batches(ds, n, seed=42):
    batch_size = 1024 * 2048
    total_tokens_needed = n * batch_size

    random.seed(seed)
    indices = list(range(len(ds)))
    random.shuffle(indices)

    collected_tokens = []
    token_count = 0

    for idx in indices:
        row = ds[idx]
        tokens = [0] + row['token_ids'] + [0] # Pad with <|endoftext|>
        collected_tokens.extend(tokens)
        token_count += len(tokens)

        if token_count >= total_tokens_needed:
            break

    # Truncate to exact size
    collected_tokens = collected_tokens[:total_tokens_needed]
    return np.array(collected_tokens)

# Load pile dataset
ds = load_dataset("pietrolesci/pile-validation", "default", split="validation", verification_mode=VerificationMode.NO_CHECKS)

for num_batches in [1, 2, 4, 8]:
    tokens = get_batches(ds, num_batches)
    print(f"Loaded first {num_batches} batches")
    for n in [1, 2, 3]:
        model = NGramModel(n, num_batches)
        model.update(tokens)
        model.cache()
        print(f"Trained {n}-gram model on {num_batches} batches")