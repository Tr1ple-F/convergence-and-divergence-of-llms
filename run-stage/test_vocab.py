import json

import numpy as np
from transformers import AutoTokenizer

model_name = "EleutherAI/pythia-12b-deduped"
revision = "step512"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    revision=revision,
    cache_dir=f"./{model_name.replace('/', '-')}/{revision}",
    bos_token='<|endoftext|>',
    eos_token='<|endoftext|>',
    add_bos_token=True,
    add_eos_token=True
)

print(tokenizer.vocab_size)

# Create vocab.json
ids = list(range(50400))
token_arr = []
for token in ids:
  token_arr.append(tokenizer.decode(token))

with open('./vocab12.json', 'w', encoding="utf8") as out_file:
  json.dump(token_arr, out_file)
