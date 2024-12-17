import sys
import numpy as np
import pandas as pd

from utils import deduped_config, strip

config = deduped_config()
revisions = config['revisions']
models = config['model_names']

data = []
for model1 in models:
    for revision1 in revisions:
        kl_data = np.load(f'../working_dir/{sys.argv[1]}/results/deduped/{model1.replace("/", "-")}-{revision1}-uni.npy')

        uniform_kl = kl_data[0]
        unigram_kl = kl_data[1]

        row = {'Model': strip(model1), 'Revision': strip(revision1), 'Uniform KL': np.mean(uniform_kl), 'Unigram KL': np.mean(unigram_kl)}
        data.append(row)

df = pd.DataFrame(data)
df.to_csv(f'../working_dir/{sys.argv[1]}/output/deduped_uni_dataframe.csv')
