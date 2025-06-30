import json
import sys
import numpy as np
from transformers import AutoTokenizer

model_name = "google/multiberts-seed_0"
revision = "step_0k"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=f"./{model_name.replace('/', '-')}/{revision}",
    add_bos_token=True,
    add_eos_token=True
)

with open(f'../working_dir/{sys.argv[1]}/input_text.txt', 'r', encoding="utf8") as file:
  input_text = file.read()

encoding = tokenizer.encode(input_text)

# Save encoded text (token ids)
np.save(f'../working_dir/{sys.argv[1]}/input_text_encoded.npy', encoding)
print(len(encoding))

# Save decoded text (text split into token parts)
decoded_arr = []
for encoded in encoding:
    decoded = tokenizer.decode(encoded)
    if (decoded == "[CLS]"):
        print(f"CLS at {len(decoded_arr)}")
    decoded_arr.append(decoded)

with open(f'../working_dir/{sys.argv[1]}/input_text_tokenized.json', 'w') as out_file:
  json.dump(decoded_arr, out_file)
