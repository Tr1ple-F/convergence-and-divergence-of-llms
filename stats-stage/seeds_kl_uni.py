import json
import sys
import os
import torch
import numpy as np
from datetime import datetime

def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def calculate_kl_divergence(log_p, log_q):
    log_p = torch.nn.functional.log_softmax(log_p, dim=-1)
    log_q = torch.nn.functional.log_softmax(log_q, dim=-1)

    return torch.sum(torch.exp(log_p) * (log_p - log_q), dim=-1)

def process_file_pair(probs1, probs2):
    divergences = calculate_kl_divergence(probs1, probs2)
    return divergences.cpu().numpy()

# Check if GPU is available, fallback to CPU if not
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open(f'../working_dir/{sys.argv[1]}/seeds_config.json', 'r') as f:
    config = json.load(f)

model_names = config['model_names']
revisions = config['revisions']
seeds = config['seeds']

uniform_base = torch.tensor(np.load('../uniform_dist.npy'), dtype=torch.float16, device=device)
unigram_base = torch.tensor(np.load('../unigram_dist.npy'), dtype=torch.float16, device=device)

for model_name_1 in model_names[int(sys.argv[2]):int(sys.argv[3])]:
    for revision_1 in revisions:
        for seed_1 in seeds:
            base_dir_1 = f'../working_dir/{sys.argv[1]}/probabilities/{model_name_1.replace("/", "-")}-seed{i}/{revision_1}'
            files_1 = [f for f in os.listdir(base_dir_1) if f.endswith('.npy')]

            all_divergences = []

            output_file = f'../working_dir/{sys.argv[1]}/results/deduped/{model_name_1.replace("/", "-")}-{revision_1}-seed{seed_1}-uni.npy'
            if os.path.exists(output_file):
                print(f"Skipping calculation for {output_file} as it already exists.")
                continue

            probs1 = torch.tensor(
                np.load(os.path.join(base_dir_1, files_1[0]))[:, :50277],
                dtype=torch.float16,
                device=device
            )

            uniform = uniform_base.expand(probs1.shape[0], -1)
            unigram = unigram_base.expand(probs1.shape[0], -1)

            print(
                f"Calculating KL divergence for model: {model_name_1} revision: {revision_1}"
            )

            all_divergences.append(
                process_file_pair(
                    probs1,
                    uniform
                )
            )
            all_divergences.append(
                process_file_pair(
                    probs1,
                    unigram
                )
            )

            np.save(output_file, np.array(all_divergences))
