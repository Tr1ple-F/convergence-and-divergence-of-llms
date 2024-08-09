import json
import os
import numpy as np

# Define the function to calculate KL divergence
def calculate_kl_divergence(p, q):
    log_p = np.array(p, dtype=np.float64)
    log_q = np.array(q, dtype=np.float64)
    return np.sum(np.exp(log_p) * (log_p - log_q))

# Load the npy files
def load_npy(file_path):
    return np.load(file_path)

# Load run_config.json
with open('kl_config.json', 'r') as f:
    config = json.load(f)

model_names = config['model_names']
revisions = config['revisions']

# Function to calculate KL divergence for a pair of files
def process_file_pair(file1_path, file2_path):
    probs1 = load_npy(file1_path)
    probs2 = load_npy(file2_path)

    # Ensure both arrays have the same shape
    assert probs1.shape == probs2.shape, (
        f"Shape mismatch between {file1_path} and {file2_path}"
    )

    # Calculate KL divergence for each timestep
    divergences = [calculate_kl_divergence(probs1[t], probs2[t]) for t in range(probs1.shape[0])]
    return divergences

# Calculate KL divergence for each combination of models and revisions
for model_name_1 in model_names:
    for revision_1 in revisions:
        base_dir_1 = f'probabilities/{model_name_1.replace("/", "-")}/{revision_1}'
        files_1 = [f for f in os.listdir(base_dir_1) if f.endswith('.npy')]

        all_divergences = []

        output_file = f'{model_name_1.replace("/", "-")}_{revision_1}_kl.npy'
        if os.path.exists(output_file):
            print(f"Skipping calculation for {output_file} as it already exists.")
            continue

        for model_name_2 in model_names:
            for revision_2 in revisions:
                # Skip comparing the same revision of the same model
                if model_name_1 == model_name_2 and revision_1 == revision_2:
                    continue

                base_dir_2 = f'probabilities/{model_name_2.replace("/", "-")}/{revision_2}'
                files_2 = [f for f in os.listdir(base_dir_2) if f.endswith('.npy')]

                # Ensure both directories have the same number of files for comparison
                assert len(files_1) == len(files_2), (
                    f"The number of files in both revisions must be the same for model {model_name_1} revision {revision_1} "
                    f"and model {model_name_2} revision {revision_2}."
                )

                print(
                    f"Calculating KL divergence between model: {model_name_1} revision: {revision_1} and model: {model_name_2} revision: {revision_2}"
                )

                all_divergences.append(
                        process_file_pair(
                            os.path.join(base_dir_1, files_1[0]),
                            os.path.join(base_dir_2, files_2[0]),
                        )
                )

        # Save the results to a npy file
        np.save(output_file, np.array(all_divergences))
