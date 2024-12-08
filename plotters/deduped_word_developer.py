import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import deduped_config, pos_tags, tagged_tokens, strip

df = pd.read_csv(f'../working_dir/{sys.argv[1]}/output/average_dataframe.csv')

df = df[df['Model 2'] == "2.8b"]
df = df[df['Revision 2'] == 143000]
df = df[df['Model 1'].isin(sys.argv[2].split(";"))]

list = ["CD", "NN", "NNPS", "VBG"]
# value_vars = [f'KL Average - {x}' for x in ["VB", "CC", "CD", "JJ", "PRP", "RP", "RB", "WP", "DT", "IN", "MD", "NN"]]
# list = ["CD", "CC", "RP", "RB", "TO", "DT", "WDT", "WP", "MD", "PRP", "PRP$", "VB", "JJ", "IN", "NN", "NNS", "NNP", "NNPS", "VBG", "VBN", "VBD", "VBZ"]

# Create a bar plot
df_melted1 = df.melt(id_vars=['Model 1', 'Model 2', 'Revision 1', 'Revision 2'], var_name='Metric', value_vars=[f'KL Average - {x}' for x in list], value_name='KL')
df_melted2 = df.melt(id_vars=['Model 1', 'Model 2', 'Revision 1', 'Revision 2'], var_name='Metric', value_vars=[f'Surprisal Average - {x}' for x in list], value_name='Surprisal')

sns.lineplot(data=df_melted1, x='Revision 1', y='KL', hue='Metric')
sns.lineplot(data=df_melted2, x='Revision 1', y='Surprisal', hue='Metric')
plt.title('KL Average by Revision and Metric')
plt.yscale('log')
plt.xscale('log')
plt.tight_layout()
plt.savefig(f'../working_dir/{sys.argv[1]}/output/analyzer_kl_{sys.argv[2].replace(";", "-")}.png')
plt.close()
