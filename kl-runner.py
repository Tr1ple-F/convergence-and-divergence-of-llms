import json
import os

import numpy as np


# Define the function to calculate KL divergence
def calculate_kl_divergence(p_probs, q_probs):
    # Convert to NumPy arrays for computation
    p_probs = np.array(p_probs)
    q_probs = np.array(q_probs)

    # Add a small value to avoid division by zero or log of zero
    epsilon = 1e-10
    p_probs = np.clip(p_probs, epsilon, 1)
    q_probs = np.clip(q_probs, epsilon, 1)

    # Calculate KL divergence
    kl_div = np.sum(p_probs * np.log(p_probs / q_probs))
    return kl_div


# Load the JSON files
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


# Load run_config.json
with open('run_config.json', 'r') as f:
    config = json.load(f)

model_names = config['model_names']
revisions = config['revisions']

json_output = []

# Calculate KL divergence for each combination of models and revisions
for model_name_1 in model_names:
    for revision_1 in revisions:
        base_dir_1 = f'probabilities/{model_name_1.replace('/', '-')}/{revision_1}'
        files_1 = [f for f in os.listdir(base_dir_1) if f.endswith('.json')]

        for model_name_2 in model_names:
            for revision_2 in revisions:
                # Skip comparing the same revision of the same model
                if model_name_1 == model_name_2 and revision_1 == revision_2:
                    continue

                base_dir_2 = f'probabilities/{model_name_2.replace('/', '-')}/{revision_2}'
                files_2 = [f for f in os.listdir(base_dir_2) if f.endswith('.json')]

                # Ensure both directories have the same number of files for comparison
                assert len(files_1) == len(
                    files_2), f"The number of files in both revisions must be the same for model {model_name_1} revision {revision_1} and model {model_name_2} revision {revision_2}."

                print(
                    f"Calculating KL divergence between model: {model_name_1} revision: {revision_1} and model: {model_name_2} revision: {revision_2}")

                # Calculate and print the KL divergence for each pair of files
                for file_name in files_1:
                    file1_path = os.path.join(base_dir_1, file_name)
                    file2_path = os.path.join(base_dir_2, file_name)

                    if not os.path.exists(file2_path):
                        print(f"File {file_name} does not exist in {base_dir_2}")
                        continue

                    probs1 = load_json(file1_path)
                    probs2 = load_json(file2_path)

                    # Extract the probabilities for common tokens
                    common_keys = set(probs1.keys()).intersection(set(probs2.keys()))

                    probs1_filtered = [probs1[key] for key in common_keys]
                    probs2_filtered = [probs2[key] for key in common_keys]

                    # Calculate KL divergence
                    kl_divergence = calculate_kl_divergence(probs1_filtered, probs2_filtered)

                    print(
                        f"KL Divergence for {file_name} between {model_name_1}/{revision_1} and {model_name_2}/{revision_2}: {kl_divergence}")

                    json_output.append({
                        "model1": model_name_1,
                        "revision1": revision_1,
                        "model2": model_name_2,
                        "revision2": revision_2,
                        "file": file_name,
                        "kl_divergence": kl_divergence
                    })

# Save the results to a JSON file
with open('kl_divergence.json', 'w') as f:
    json.dump(json_output, f, indent=4)
