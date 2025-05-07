import sys
import numpy as np
import pandas as pd
from utils import seeds_config, strip, tagged_tokens, pos_tags

config = seeds_config()
revisions = config['revisions']
models = config['model_names'][2:5]
seeds = config['seeds']

data = []
for model1 in models:
    for revision1 in revisions:
        kl_data = np.load(f'../working_dir/{sys.argv[1]}/results/cross/{model1.replace("/", "-")}-{revision1}-kl.npy')

        pos_kl = []
        for i, pos_tag in enumerate(pos_tags()):
            pos_indices = [idx for idx, tag in enumerate(tagged_tokens()) if tag[1] == pos_tag["tag"]]
            surprisal_averages = np.mean(kl_data[:, pos_indices], axis = 1)
            pos_kl.append(surprisal_averages)

        overall_average = np.mean(kl_data, axis = 1)

        i = 0
        for seed in seeds:
            row = {'Model': strip(model1), 'Revision': strip(revision1), 'Seed': seed, 'KL Average': overall_average[i]}

            for j, pos_tag in enumerate(pos_tags()):
                row[f'KL Average - {pos_tag["tag"]}'] = pos_kl[j][i]

            data.append(row)
            i += 1

        print(f'Finished {model1} {revision1}')

df = pd.DataFrame(data)
df.to_csv(f'../working_dir/{sys.argv[1]}/output/cross_dataframe.csv')
