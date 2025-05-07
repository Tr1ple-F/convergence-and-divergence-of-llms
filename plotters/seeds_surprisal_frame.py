import sys
import numpy as np
import pandas as pd
from utils import seeds_config, strip, tagged_tokens, pos_tags

config = seeds_config()
revisions = config['revisions']
models = config['model_names']
seeds = config['seeds']

data = []
for model1 in models:
    for revision1 in revisions:
        print(revision1)
        for i in seeds:
            surprisal_data = np.load(f'../working_dir/{sys.argv[1]}/results/seeds/{model1.replace("/", "-")}-{revision1}-seed{i}-surprisal.npy')

            pos_surprisal = []
            for i, pos_tag in enumerate(pos_tags()):
                pos_indices = [idx for idx, tag in enumerate(tagged_tokens()) if tag[1] == pos_tag["tag"]]
                surprisal_averages = np.mean(surprisal_data[pos_indices])
                pos_surprisal.append(surprisal_averages)

            overall_average = np.mean(surprisal_data)

            row = {'Model': strip(model1), 'Revision': strip(revision1), 'Seed': i, 'Surprisal': overall_average}

            for j, pos_tag in enumerate(pos_tags()):
                row[f'Surprisal Average - {pos_tag["tag"]}'] = pos_surprisal[j]

            data.append(row)

df = pd.DataFrame(data)
df.to_csv(f'../working_dir/{sys.argv[1]}/output/seeds_surprisal_dataframe.csv')
