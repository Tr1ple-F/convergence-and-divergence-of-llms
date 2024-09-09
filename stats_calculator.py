import json

import numpy as np

def calculate_surprisal(probabilities, correct_indices):
    return -probabilities[np.arange(probabilities.shape[0]), correct_indices]

def calculate_stats(model_name, revision, correct_indices, top_n=10):
    probabilities = np.load(f'probabilities/{model_name.replace("/", "-")}/{revision}/probabilities.npy')

    # Calculate surprisal
    surprisal = calculate_surprisal(probabilities, correct_indices[1:])

    # Get the top N tokens with the highest probabilities for each token
    top_tokens = np.argsort(probabilities, axis=-1)[:, -top_n:]
    top_probabilities = np.take_along_axis(probabilities, top_tokens, axis=-1)

    # Save as npy files
    path1 = f'results/{model_name.replace("/", "-")}-{revision}-surprisal.npy'
    path2 = f'results/{model_name.replace("/", "-")}-{revision}-top_tokens.npy'
    path3 = f'results/{model_name.replace("/", "-")}-{revision}-top_probabilities.npy'
    np.save(path1, surprisal)
    np.save(path2, top_tokens)
    np.save(path3, top_probabilities)

with open('kl_config.json', 'r') as f:
    config = json.load(f)

model_names = config['model_names']
revisions = config['revisions']

correct_indices_file = 'input_text_encoded.npy'
correct_indices = np.load(correct_indices_file)

for model_name in model_names:
    for revision in revisions:
        calculate_stats(model_name, revision, correct_indices)
