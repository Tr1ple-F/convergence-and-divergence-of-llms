import sys
import numpy as np
import pandas as pd
from utils import seeds_config, strip, tagged_tokens

config = seeds_config()
revisions = config['revisions']
models = config['model_names']
seeds = config['seeds']

correct_indices = np.load(f'../working_dir/{sys.argv[1]}/input_text_encoded.npy')
data = []

for model1 in models:
    surprisal_data = np.load(f'../working_dir/{sys.argv[1]}/results/seeds/{model1.replace("/", "-")}-{revisions[-1]}-seed1-surprisal.npy')
    for revision1 in revisions:
        for seed1 in seeds:
            kl_data = np.load(f'../working_dir/{sys.argv[1]}/results/seeds/{model1.replace("/", "-")}-{revision1}-seed{seed1}-kl.npy')

            i = 0
            for model2 in models:
                for revision2 in revisions:
                    for seed2 in seeds:
                        if model1 == model2 and revision1 == revision2:
                            for x in range(0, len(correct_indices) - 1):
                                row = {'Model': strip(model1), 'Training Step': strip(revision1), "Model 2": strip(model2), "Training Step 2": strip(revision2), 'Seed 1': seed1, 'Seed 2': seed2, 'KL': kl_data[i, x], 'Surprisal': surprisal_data[x]}
                                data.append(row)

                        i += 1

            print(f"Finished {model1} {revision1} {seed1}")

df = pd.DataFrame(data)
df.to_csv(f'../working_dir/{sys.argv[1]}/output/seeds_surprisal_by_token.csv')
