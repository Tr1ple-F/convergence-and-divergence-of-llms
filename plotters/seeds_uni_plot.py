import matplotlib.pyplot as plt
import pandas as pd
import sys
import seaborn as sns

df = pd.read_csv(f'../working_dir/{sys.argv[1]}/output/dataframes/seeds_uni_dist.csv')

df_uniform = df[['Revision', 'Model', 'Uniform KL']].rename(columns={'Uniform KL': 'KL'})
df_uniform['Distribution'] = 'Uniform'
df_unigram = df[['Revision', 'Model', 'Unigram KL']].rename(columns={'Unigram KL': 'KL'})
df_unigram['Distribution'] = 'Unigram'
df_out = pd.concat([df_uniform, df_unigram])

plt.figure(figsize=(10, 6))
sns.lineplot(data=df_out, x='Revision', y='KL', hue='Model', errorbar='sd', style="Distribution")
plt.xscale('log')
plt.savefig(f'../working_dir/{sys.argv[1]}/output/seeds_uni_dist.png')
plt.close()
