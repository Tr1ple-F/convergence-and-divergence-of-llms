import json
import os
import numpy as np

def calculate_kl_divergence(log_p, log_q):
    return np.sum(np.exp(log_p) * (log_p - log_q), axis=-1)

def process_file_pair(file1_path, file2_path):
    probs1 = np.load(file1_path)
    probs2 = np.load(file2_path)

    assert probs1.shape == probs2.shape, (
        f"Shape mismatch between {file1_path} and {file2_path}"
    )

    divergences = calculate_kl_divergence(probs1, probs2)
    return divergences


with open('seeds_config.json', 'r') as f:
    config = json.load(f)

model_names = config['model_names']
revisions = config['revisions']

for model_name_1 in model_names:
    for revision_1 in revisions:
        for i in [1,3,5,7,9]:
            base_dir_1 = f'../probabilities/{model_name_1.replace("/", "-")}-seed{i}/{revision_1}'
            files_1 = [f for f in os.listdir(base_dir_1) if f.endswith('.npy')]

            all_divergences = []

            output_file = f'../results/seeds/{model_name_1.replace("/", "-")}-{revision_1}-kl.npy'
            if os.path.exists(output_file):
                print(f"Skipping calculation for {output_file} as it already exists.")
                continue

            for model_name_2 in model_names:
                for revision_2 in revisions:
                    for j in [1,3,5,7,9]:
                        base_dir_2 = f'../probabilities/{model_name_2.replace("/", "-")}-seed{j}/{revision_2}'
                        files_2 = [f for f in os.listdir(base_dir_2) if f.endswith('.npy')]

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
                        )  # Model 1 revision 1 is P and Model 2 revision 2 is Q

            # Save the results to a npy file
            np.save(output_file, np.array(all_divergences))
