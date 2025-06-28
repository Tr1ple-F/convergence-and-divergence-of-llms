import json
import sys
import numpy as np
import os

def calculate_surprisal(probabilities, correct_indices):
    return -probabilities[np.arange(probabilities.shape[0]), correct_indices]

def calculate_stats(model_name, revision, correct_indices, top_n=10):
    for i in seeds:
        path1 = f'../working_dir/{sys.argv[1]}/results/seeds/{model_name.replace("/", "-")}-{revision}-seed{i}-surprisal.npy'

        if os.path.exists(path1):
            print(f"Skipping {model_name.replace('/', '-')}-{revision}-seed{i}")
            continue

        probabilities = np.load(f'../working_dir/{sys.argv[1]}/probabilities/{model_name.replace("/", "-")}-seed_{i}/{revision}/probabilities.npy')[:, :30522]

        # Calculate surprisal
        surprisal = calculate_surprisal(probabilities, correct_indices[1:])

        # Save as npy files
        np.save(path1, surprisal)
        print(f"Finished processing {model_name.replace('/', '-')}-{revision}-seed{i}")

with open(f'../working_dir/{sys.argv[1]}/seeds_config.json', 'r') as f:
    config = json.load(f)

model_names = config['model_names']
revisions = config['revisions']
seeds = config['seeds']

correct_indices_file = f'../working_dir/{sys.argv[1]}/input_text_encoded.npy'
correct_indices = np.load(correct_indices_file)

for model_name in model_names:
    for revision in revisions:
        calculate_stats(model_name, revision, correct_indices)
