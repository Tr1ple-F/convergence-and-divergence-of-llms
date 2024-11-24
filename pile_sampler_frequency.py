from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

print("Loading dataset...")
ds = load_dataset("pietrolesci/pile-validation")
print("Dataset loaded.")

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

token_counter = np.ndarray(len(tokenizer))

i = 0
for x in range(len(ds['validation'])):
    if i % 1000 == 0:
        print(f'{i} / 214670 done')
    sample_text = ds['validation'][x]['text']
    out = tokenizer.encode(sample_text)
    for y in out:
        token_counter[y] += 1
    i += 1

np.save("./frequency_count.npy", token_counter)
