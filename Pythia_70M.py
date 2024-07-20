from transformers import GPTNeoXForCausalLM, AutoTokenizer
import torch
import numpy as np

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

# inputs = tokenizer("This is the text I want to predict on", return_tensors="pt")
# output = model(**inputs)
# logits = output.logits

# probabilities = torch.nn.functional.softmax(logits, dim=-1)
# print(logits)
# print("Predictions: ", probabilities)

input_text = "Today is"
inputs = tokenizer([input_text], return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50, return_dict_in_generate=True, output_scores=True)
transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)

input_length = inputs.input_ids.shape[1]
generated_tokens = outputs.sequences[:, input_length:]

for tok, score in zip(generated_tokens[0], transition_scores[0]):
    print(f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score.numpy():.4f} | {np.exp(score.numpy()):.2%}")
