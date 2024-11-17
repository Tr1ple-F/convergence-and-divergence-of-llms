import numpy as np
from transformers import GPTNeoXForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import os
import json
import sys

def get_probabilities(model_name, revision, input_text):
    cache_dir = f"./{model_name.replace('/', '-')}/{revision}"

    model = GPTNeoXForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        cache_dir=cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-14m-seed1",
        revision="step128",
        cache_dir=f"./EleutherAI-pythia-14m-seed1/step128",
        bos_token = '<|endoftext|>',
        eos_token = '<|endoftext|>',
        add_bos_token = True,
        add_eos_token = True
    )

    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    output_dir = os.path.join(f"../working_dir/{sys.argv[1]}/probabilities/" + model_name.replace('/', '-'), revision)
    os.makedirs(output_dir, exist_ok=True)

    inputs = tokenizer(input_text, return_tensors="pt")

    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Context window settings
    window_size = 2048
    overlap = 1024
    stride = window_size - overlap

    all_probabilities = []

    with torch.no_grad():
        for start_idx in range(0, inputs["input_ids"].size(1) - overlap, stride):
            end_idx = start_idx + window_size
            window_inputs = {key: value[:, start_idx:end_idx] for key, value in inputs.items()}

            logits = model(**window_inputs).logits
            probabilities = F.log_softmax(logits, dim=-1).cpu().numpy().reshape(-1, logits.shape[-1])
            if start_idx == 0:
                # Append the whole window for the first case
                all_probabilities.append(probabilities)
            else:
                # Append only the new part past the overlap
                all_probabilities.append(probabilities[overlap:])

    # Flatten the list of arrays into a single array
    all_probabilities_matrix = np.concatenate(all_probabilities, axis=0)
    all_probabilities_matrix = all_probabilities_matrix[:-1]

    output_file_path = os.path.join(output_dir, "probabilities.npy")
    np.save(output_file_path, all_probabilities_matrix.astype(np.float16))

def main():
    with open(f'../working_dir/{sys.argv[1]}/seeds_config.json', 'r') as config_file:
        config = json.load(config_file)

    model_names = config['model_names']
    revisions = config['revisions']

    with open(f'../working_dir/{sys.argv[1]}/input_text.txt', 'r', encoding="utf8") as file:
        input_text = file.read()

        for model_name in model_names:
            for revision in revisions:
                for i in [1,5,9]:
                    print(f"Processing model: {model_name}-seed{i}, revision: {revision}")
                    get_probabilities(f"{model_name}-seed{i}", revision, input_text)


if __name__ == "__main__":
    main()
