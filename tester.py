import numpy as np
from transformers import GPTNeoXForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import os
from shutil import rmtree
import json

def generate_top_probabilities(model_name, revision, input_text_file):
    # Load the model and tokenizer
    model = GPTNeoXForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        cache_dir=f"./{model_name.replace('/', '-')}/{revision}",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        cache_dir=f"./{model_name.replace('/', '-')}/{revision}",
    )

    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    # Read the input text file
    with open(input_text_file, 'r', encoding="utf8") as file:
        input_text = file.read()

    # Ensure the output directory exists
    output_dir = os.path.join("probabilities/" + model_name.replace('/', '-'), revision)
    os.makedirs(output_dir, exist_ok=True)

    # Tokenize the current text
    inputs = tokenizer(input_text, return_tensors="pt")

    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Get the logits from the model
    window_size = 2048
    overlap = 1024
    stride = window_size - overlap

    all_probabilities = []

    with torch.no_grad():
        for start_idx in range(0, inputs.input_ids.size(1) - window_size + 1, stride):
            end_idx = start_idx + window_size
            window_inputs = {key: value[:, start_idx:end_idx] for key, value in inputs.items()}

            logits = model(**window_inputs).logits
            probabilities = F.log_softmax(logits, dim=-1)
            all_probabilities.append(probabilities.reshape(-1, probabilities.shape[-1]).cpu().numpy())

    # Concatenate all probability matrices
    all_probabilities_matrix = np.concatenate(all_probabilities, axis=0)

    # Define the output file path
    output_file_path = os.path.join(output_dir, "probabilities.npy")

    # Save the NumPy array to a file
    np.save(output_file_path, all_probabilities_matrix)

def delete_cache_directory(model_name, revision):
    cache_dir = f"./{model_name.replace('/', '-')}/{revision}"
    if os.path.exists(cache_dir):
        rmtree(cache_dir)
        print(f"Deleted cache directory: {cache_dir}")

def main():
    # Read the configuration from run_config.json
    with open('run_config.json', 'r') as config_file:
        config = json.load(config_file)

    # Extract model names, revisions, and input text file from the configuration
    model_names = config['model_names']
    revisions = config['revisions']
    input_text_file = config['input_text_file']

    # Iterate over all combinations of model names and revisions
    for model_name in model_names:
        for revision in revisions:
            print(f"Processing model: {model_name}, revision: {revision}")
            generate_top_probabilities(model_name, revision, input_text_file)
            delete_cache_directory(model_name, revision)


if __name__ == "__main__":
    main()
