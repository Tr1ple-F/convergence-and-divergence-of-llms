import sys
import numpy as np
import pandas as pd
from utils import seeds_config, strip

config = seeds_config()
revisions = config['revisions']
models = config['model_names']
seeds = config['seeds']

frequency_data = np.load(f'../frequency_count.npy')
correct_indices = np.load(f'../working_dir/{sys.argv[1]}/input_text_encoded.npy')
data = []

for model1 in models:
    for revision1 in revisions:
        for seed1 in seeds:
            kl_data = np.load(f'../working_dir/{sys.argv[1]}/results/seeds/{model1.replace("/", "-")}-{revision1}-seed{seed1}-kl.npy')

            # print(kl_data.shape)

            i = 0
            for model2 in models:
                for revision2 in revisions:
                    for seed2 in seeds:
                        if model1 == model2 and revision1 == revision2:
                            for x in range(0, len(correct_indices) - 1):
                                row = {'Model 1': strip(model1), 'Revision 1': strip(revision1), "Model 2": strip(model1), "Revision 2": strip(revision1), 'Seed 1': seed1, 'Seed 2': seed2, 'KL': kl_data[i, x], 'Frequency': frequency_data[correct_indices[x + 1]]}
                                data.append(row)

                        i += 1

            print(f"Finished {model1} {revision1} {seed1}")

df = pd.DataFrame(data)
print(df)
df.to_csv(f'../working_dir/{sys.argv[1]}/output/seeds_frequency_dataframe.csv')
