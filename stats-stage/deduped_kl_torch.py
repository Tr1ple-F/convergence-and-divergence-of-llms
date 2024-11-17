import json
import os
import torch
import numpy as np
from datetime import datetime

def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def calculate_kl_divergence(log_p, log_q):
    log_p = torch.nn.functional.log_softmax(log_p, dim=-1)
    log_q = torch.nn.functional.log_softmax(log_q, dim=-1)

    # Calculate KL divergence
    return torch.sum(torch.exp(log_p) * (log_p - log_q), dim=-1)

def process_file_pair(probs1, file2_path, device):
    probs2 = torch.tensor(np.load(file2_path)[:, :50254], dtype=torch.float16, device=device)
    assert probs1.shape == probs2.shape, (
        f"Shape mismatch between {file2_path}"
    )
    divergences = calculate_kl_divergence(probs1, probs2)
    return divergences.cpu().numpy()

# Check if GPU is available, fallback to CPU if not
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('deduped_config.json', 'r') as f:
    config = json.load(f)

model_names = config['model_names']
revisions = config['revisions']

for model_name_1 in model_names:
    for revision_1 in revisions:
        base_dir_1 = f'../probabilities/{model_name_1.replace("/", "-")}/{revision_1}'
        files_1 = [f for f in os.listdir(base_dir_1) if f.endswith('.npy')]

        all_divergences = []

        output_file = f'../results/deduped/{model_name_1.replace("/", "-")}-{revision_1}-kl.npy'
        if os.path.exists(output_file):
            print(f"Skipping calculation for {output_file} as it already exists.")
            continue

        probs1 = torch.tensor(
            np.load(os.path.join(base_dir_1, files_1[0]))[:, :50254],
            dtype=torch.float16,
            device=device
        )

        print(
            f"Calculating KL divergence for model: {model_name_1} revision: {revision_1}"
        )

        for model_name_2 in model_names:
            for revision_2 in revisions:
                base_dir_2 = f'../probabilities/{model_name_2.replace("/", "-")}/{revision_2}'
                files_2 = [f for f in os.listdir(base_dir_2) if f.endswith('.npy')]

                assert len(files_1) == len(files_2), (
                    f"The number of files in both revisions must be the same for model {model_name_1} revision {revision_1} "
                    f"and model {model_name_2} revision {revision_2}."
                )

                all_divergences.append(
                    process_file_pair(
                        probs1,
                        os.path.join(base_dir_2, files_2[0]),
                        device=device
                    )
                )

        np.save(output_file, np.array(all_divergences))
