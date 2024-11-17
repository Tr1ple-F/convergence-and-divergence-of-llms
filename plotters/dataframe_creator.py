import json
import numpy as np
import pandas as pd
from utils import deduped_config

with open('../common/pos_tagged_tokens.json', 'r') as f:
    pos_tagged_tokens = json.load(f)

with open('../common/pos_tags.json', 'r') as f:
    unique_pos_tags = json.load(f)

config = deduped_config()
revisions = config['revisions']
models = config['model_names']

data = []
for model1 in models:
    for revision1 in revisions:
        kl_data = np.load(f'../results/deduped/{model1.replace("/", "-")}-{revision1}-kl.npy')
        surprisal_data = np.load(f'../results/deduped/{model1.replace("/", "-")}-{revision1}-surprisal.npy')

        pos_kl = []
        pos_surprisal = []
        for i, pos_tag in enumerate(unique_pos_tags):
            pos_indices = [idx for idx, tag in enumerate(pos_tagged_tokens) if tag[1] == pos_tag["tag"]]
            kl_averages = np.mean(kl_data[:, pos_indices], axis=1)
            pos_kl.append(kl_averages)
            surprisal_averages = np.mean(surprisal_data[:, pos_indices], axis=1)
            pos_surprisal.append(surprisal_averages)

        overall_average = np.mean(kl_data, axis =  1)

        i = 0
        for model2 in models:
            for revision2 in models:
                row = {'Model 1': model1, 'Revision 1': revision1, "Model 2": model2, "Revision 2": revision2, 'KL Average': overall_average[i], 'Surprisal Average': np.mean(surprisal_data[i])}

                for j, pos_tag in enumerate(unique_pos_tags):
                    row[f'KL Average - {pos_tag["tag"]}'] = pos_kl[j][i]
                    row[f'Surprisal Average - {pos_tag["tag"]}'] = pos_surprisal[j][i]

                data.append(row)
                i += 1

df = pd.DataFrame(data)
df.to_csv('../graphics/average_dataframe.csv')
