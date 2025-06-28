from datasets import load_dataset, VerificationMode
from transformers import AutoTokenizer
import numpy as np

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

token_counter = np.ndarray(len(tokenizer))

i = 0
for x in range(len(ds)):
    if i % 1000 == 0:
        print(f'{i} / 214670 done')
    sample_text = ds[x]['text']
    out = tokenizer.encode(sample_text)
    for y in out:
        token_counter[y] += 1
    i += 1

np.save("./frequency_count_bert.npy", token_counter)
