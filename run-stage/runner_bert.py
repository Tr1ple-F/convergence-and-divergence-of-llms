import numpy as np
from requests.models import ChunkedEncodingError
from transformers import AutoTokenizer, AutoModelForPreTraining
import torch
import torch.nn.functional as F
import os
import json
import sys

def get_probabilities(model_name, revision, input_text):
    output_dir = os.path.join(f"../working_dir/{sys.argv[1]}/probabilities/" + model_name.replace('/', '-'), revision)
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, "probabilities.npy")
    if (os.path.exists(output_file_path)):
        print(f"Probabilities for model {model_name}, revision {revision} already exist. Skipping.")
        return

    cache_dir = f"./{model_name.replace('/', '-')}/{revision}"

    model = AutoModelForPreTraining.from_pretrained(
        f"{model_name}-{revision}",
        cache_dir=cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        f"{model_name}-{revision}",
        cache_dir = cache_dir,
        add_bos_token=True,
        add_eos_token=True
    )

    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    inputs = []
    for snippet in input_text.split("[SEP][CLS]"):
        inputs.append(tokenizer(snippet, return_tensors="pt"))

    inputs_loaded = []
    if torch.cuda.is_available():
        for snippet in inputs:
            inputs_loaded.append({k: v.cuda() for k, v in snippet.items()})

    all_probabilities = []
    with torch.no_grad():
        for snippet in inputs_loaded:
            failed = True
            while failed:
                try:
                    window_inputs = snippet
                    logits = model(**window_inputs).prediction_logits # ?
                    probabilities = F.log_softmax(logits, dim=-1).cpu().numpy().reshape(-1, logits.shape[-1])
                    all_probabilities.append(probabilities)
                    failed = False
                except ChunkedEncodingError:
                    continue

    # Flatten the list of arrays into a single array
    all_probabilities_matrix = np.concatenate(all_probabilities, axis=0)
    all_probabilities_matrix = all_probabilities_matrix[:-1] # Because of last [SEP]

    np.save(output_file_path, all_probabilities_matrix.astype(np.float16))

def main():
    with open(f'../working_dir/{sys.argv[1]}/seeds_config.json', 'r') as config_file:
        config = json.load(config_file)

    model_names = config['model_names']
    revisions = config['revisions']
    seeds = sys.argv[2].split(',')

    with open(f'../working_dir/{sys.argv[1]}/input_text.txt', 'r', encoding="utf8") as file:
        input_text = file.read()

        for model_name in model_names:
            for revision in revisions:
                for i in seeds:
                    print(f"Processing model: {model_name}-seed_{i}, revision: {revision}")
                    get_probabilities(f"{model_name}-seed_{i}", revision, input_text)


if __name__ == "__main__":
    main()
