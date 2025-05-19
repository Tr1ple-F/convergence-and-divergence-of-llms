import sys

import ipdb
import numpy as np
import pandas as pd
from utils import seeds_config, strip

config = seeds_config()
revisions = config['revisions']
models = config['model_names']
seeds = config['seeds']

correct_indices = np.load(f'../working_dir/{sys.argv[1]}/input_text_encoded.npy')
data = []

for model1 in models:
    for revision1 in revisions:
        for seed1 in seeds:
            surprisal_data = np.load(f'../working_dir/{sys.argv[1]}/results/seeds/{model1.replace("/", "-")}-{revision1}-seed{seed1}-surprisal.npy')

            row = {'Model': strip(model1), 'Training Step': strip(revision1), 'Seed': seed1, 'ICL': surprisal_data[500]-surprisal_data[50]}
            data.append(row)

df = pd.DataFrame(data)
df.to_csv(f'../working_dir/{sys.argv[1]}/output/seeds_icl.csv')
