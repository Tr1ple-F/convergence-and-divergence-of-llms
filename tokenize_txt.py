import json

import numpy as np
from transformers import AutoTokenizer

model_name = "EleutherAI/pythia-70m-deduped"
revision = "step512"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    revision=revision,
    cache_dir=f"./{model_name.replace('/', '-')}/{revision}",
)

with open("input_text.txt", 'r', encoding="utf8") as file:
    input_text = file.read()

encoding = tokenizer.encode(input_text)

ids = list(range(50304))

token_arr = []
for token in ids:
    token_arr.append(tokenizer.decode(token))

with open('vocab.json', 'w', encoding="utf8") as out_file:
    json.dump(token_arr, out_file)

np.save('input_text_encoded.npy', encoding)

decoded_arr = []
for token in encoding:
    decoded_arr.append(tokenizer.decode(token))

with open('input_text_tokenized.json', 'w') as out_file:
    json.dump(decoded_arr, out_file)
