import json

import numpy as np
from transformers import AutoTokenizer

model_name = "EleutherAI/pythia-70m-deduped"
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

with open("common/input_text.txt", 'r', encoding="utf8") as file:
  input_text = file.read()

encoding = tokenizer.encode(input_text)

# Create vocab.json
ids = list(range(50304))
token_arr = []
for token in ids:
  token_arr.append(tokenizer.decode(token))

with open('common/vocab.json', 'w', encoding="utf8") as out_file:
  json.dump(token_arr, out_file)

# Save encoded text (token ids)
np.save('common/input_text_encoded.npy', encoding)

# Save decoded text (text split into token parts)
decoded_arr = []
for token in encoding:
  decoded_arr.append(tokenizer.decode(token))

with open('common/input_text_tokenized.json', 'w') as out_file:
  json.dump(decoded_arr, out_file)
