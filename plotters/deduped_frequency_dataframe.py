import sys
import numpy as np
import pandas as pd
from utils import deduped_config, strip

config = deduped_config()
revisions = config['revisions']
models = config['model_names']

frequency_data = np.load(f'../frequency_count.npy')
correct_indices = np.load(f'../working_dir/{sys.argv[1]}/input_text_encoded.npy')
data = []

for model1 in models:
    for revision1 in revisions:
        kl_data = np.load(f'../working_dir/{sys.argv[1]}/results/deduped/{model1.replace("/", "-")}-{revision1}-kl.npy')

        i = 0
        for model2 in models:
            for revision2 in revisions:
                for x in range(0, len(correct_indices) - 1):
                    row = {'Model 1': strip(model1), 'Revision 1': strip(revision1), "Model 2": strip(model2), "Revision 2": strip(revision2), 'KL': kl_data[i, x], 'Frequency': frequency_data[correct_indices[x + 1]]}
                    data.append(row)

                i += 1

        print(f"Finished {model1} {revision1}")

df = pd.DataFrame(data)
df.to_csv(f'../working_dir/{sys.argv[1]}/output/frequency_dataframe.csv')
