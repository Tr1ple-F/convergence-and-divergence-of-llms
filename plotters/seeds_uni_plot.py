import matplotlib.pyplot as plt
import pandas as pd
import sys
import seaborn as sns

df = pd.read_csv(f'../working_dir/{sys.argv[1]}/output/seeds_uni_dataframe.csv')

df_uniform = df[['Training Step', 'Model', 'Uniform KL']].rename(columns={'Uniform KL': 'KL'})
df_uniform['Distribution'] = 'Uniform'
df_unigram = df[['Training Step', 'Model', 'Unigram KL']].rename(columns={'Unigram KL': 'KL'})
df_unigram['Distribution'] = 'Unigram'
df_out = pd.concat([df_uniform, df_unigram])

plt.figure(figsize=(10, 6))
sns.lineplot(data=df_out, x='Training Step', y='KL', hue='Model', errorbar='ci', style="Distribution")
plt.xscale('log')
plt.savefig(f'../working_dir/{sys.argv[1]}/output/seeds_uni_dist.png')
plt.close()
