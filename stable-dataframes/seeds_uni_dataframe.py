import sys
import numpy as np
import pandas as pd

from utils import seeds_config, strip

config = seeds_config()
revisions = config['revisions']
models = config['model_names']
seeds = config['seeds']

data = []
for model1 in models:
    for revision1 in revisions:
        for seed1 in seeds:
            kl_data = np.load(f'../working_dir/{sys.argv[1]}/results/seeds/{model1.replace("/", "-")}-{revision1}-seed{seed1}-uni.npy')

            uniform_kl = kl_data[0]
            unigram_kl = kl_data[1]

            row = {'Model': strip(model1), 'Revision': strip(revision1), 'Seed': seed1, 'Uniform KL': np.mean(uniform_kl), 'Unigram KL': np.mean(unigram_kl)}
            data.append(row)

df = pd.DataFrame(data)
df.to_csv(f'../working_dir/{sys.argv[1]}/output/seeds_uni_dataframe.csv')
