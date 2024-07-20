from transformers import GPTNeoXForCausalLM, AutoTokenizer
import torch

model = GPTNeoXForCausalLM.from_pretrained(
    "EleutherAI/pythia-70m-deduped",
    revision="step143000",
    cache_dir="./pythia-70m-deduped/step143000",
)

tokenizer = AutoTokenizer.from_pretrained(
    "EleutherAI/pythia-70m-deduped",
    revision="step143000",
    cache_dir="./pythia-70m-deduped/step143000",
)

model.eval()

if torch.cuda.is_available():
    model = model.cuda()

inputs = tokenizer("This is the text I want to predict on", return_tensors="pt")
outputs = model()

prediction_logits = outputs.logits
print(prediction_logits)
