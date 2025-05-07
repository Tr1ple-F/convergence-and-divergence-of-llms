import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import deduped_config, pos_tags, tagged_tokens, strip

df = pd.read_csv(f'../working_dir/{sys.argv[1]}/output/deduped_pos_current_dataframe.csv')

# Rename column revision 1 to training step and surprisal average to cross entropy
df = df.rename(columns={'Surprisal Average': 'Cross-Entropy'})

sns.lineplot(data=df, x='Training Step', y=f'Cross-Entropy', hue='Model')
plt.title('Cross-Entropy by Training Step')
plt.xscale('log')
plt.tight_layout()
plt.savefig(f'../working_dir/{sys.argv[1]}/output/model_cross_entropy.png')
plt.close()
