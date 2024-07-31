
from transformers import AutoTokenizer

model_name = "EleutherAI/pythia-160m-deduped"
revision = "step512"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    revision=revision,
    cache_dir=f"./{model_name.replace('/', '-')}/{revision}",
)

encoding = tokenizer.encode("hello test text")
print(encoding)
decoding = tokenizer.decode(encoding)
print(type(decoding))
