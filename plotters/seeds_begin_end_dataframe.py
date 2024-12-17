import sys
import numpy as np
import pandas as pd
from utils import seeds_config, strip

config = seeds_config()
revisions = config['revisions']
models = config['model_names']
seeds = config['seeds']

begin_indices = np.load(f'../working_dir/{sys.argv[1]}/begin_tokens.npy')
end_indices = np.load(f'../working_dir/{sys.argv[1]}/end_tokens.npy')

data = []
for model1 in models:
    for revision1 in revisions:
        for seed1 in seeds:
            kl_data = np.load(f'../working_dir/{sys.argv[1]}/results/seeds/{model1.replace("/", "-")}-{revision1}-seed{seed1}-kl.npy')
            surprisal_data = np.load(f'../working_dir/{sys.argv[1]}/results/seeds/{model1.replace("/", "-")}-{revision1}-seed{seed1}-surprisal.npy')

            begin_averages = np.mean(kl_data[:, begin_indices[]], axis=1)
            end_averages = np.mean(kl_data[:, end_indices], axis=1)
            total_averages = np.mean(kl_data, axis = 1)

            begin_surprisal = np.mean(surprisal_data[begin_indices])
            end_surprisal = np.mean(surprisal_data[end_indices])
            total_surprisal = np.mean(surprisal_data)

            i = 0
            for model2 in models:
                for revision2 in revisions:
                    for seed2 in seeds:
                        row = {
                            'Model 1': strip(model1),
                            'Revision 1': strip(revision1),
                            'Seed 1': seed1,
                            "Model 2": strip(model2),
                            "Revision 2": strip(revision2),
                            "Seed 2": seed2,
                            'KL Average': total_averages[i],
                            'Surprisal Average': total_surprisal,
                            'Begin KL Average': begin_averages[i],
                            'Begin Surprisal Average': begin_surprisal,
                            'End KL Average': end_averages[i],
                            'End Surprisal Average': end_surprisal,
                            'Begin Ratio KL': begin_averages[i] / total_averages[i],
                            'End Ratio KL': end_averages[i] / total_averages[i],
                        }

                        data.append(row)
                        i += 1

df = pd.DataFrame(data)
df.to_csv(f'../working_dir/{sys.argv[1]}/output/seed_begin_end_dataframe.csv')
