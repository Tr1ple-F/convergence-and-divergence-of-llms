import json
import sys
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

with open(f'../working_dir/{sys.argv[1]}/seeds_config.json', 'r', encoding="utf8") as file:
  input_text = file.read()

encoding = tokenizer.encode(input_text)

# Save encoded text (token ids)
np.save(f'../working_dir/{sys.argv[1]}/seeds_config.json', encoding)
print(len(encoding))

# Save decoded text (text split into token parts)
decoded_arr = []
for token in encoding:
  decoded_arr.append(tokenizer.decode(token))

with open(f'../working_dir/{sys.argv[1]}/seeds_config.json', 'w') as out_file:
  json.dump(decoded_arr, out_file)
