from transformers import GPTNeoXForCausalLM, AutoTokenizer
import torch
import numpy as np
import torch.nn.functional as F
import os
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
    with open(input_text_file, 'r') as file:
        input_text = file.read()

    # Split input text into words
    words = input_text.split()

    # Initialize progressive input text
    progressive_text = ""

    # Ensure the output directory exists
    output_dir = os.path.join("probabilities/" + model_name.replace('/', '-'), revision)
    os.makedirs(output_dir, exist_ok=True)

    # Process each word in the input text
    for idx, word in enumerate(words):
        if progressive_text:
            progressive_text += " "
        progressive_text += word

        # Tokenize the current progressive text
        inputs = tokenizer([progressive_text], return_tensors="pt")

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Get the logits from the model
        with torch.no_grad():
            logits = model(**inputs).logits

        # Get the logits for the last token in the input
        last_token_logits = logits[:, -1, :]  # Reduces size from 1x2x50304 to 1x50304

        # Apply softmax to get probabilities
        probabilities = F.softmax(last_token_logits, dim=-1)

        # Convert to CPU and NumPy array for easier handling
        probabilities = probabilities.cpu().numpy()

        # Get the top k token probabilities
        top_k = 10
        top_k_indices = np.argsort(probabilities[0])[::-1][:top_k]

        # Create a dictionary for the top k tokens and their probabilities
        top_k_probs = {tokenizer.decode([idx]): float(probabilities[0][idx]) for idx in top_k_indices}

        # Define the output file path
        output_file_path = os.path.join(output_dir, f"{idx + 1}.json")

        # Save the dictionary to a JSON file
        with open(output_file_path, 'w') as json_file:
            json.dump(top_k_probs, json_file, indent=4)


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


if __name__ == "__main__":
    main()
