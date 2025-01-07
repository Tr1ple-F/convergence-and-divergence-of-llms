import matplotlib.pyplot as plt
import pandas as pd
import sys
import seaborn as sns

df = pd.read_csv(f'../working_dir/{sys.argv[1]}/output/dist_average_dataframe.csv')
# df = df[df['Model'].isin(['12b','6.9b', '2.8b'])]
# df = df[df['Model'] != '12b']
# df = df[df['Model'] != '6.9b']

df_uniform = df[['Training Step', 'Model', 'Uniform KL']].rename(columns={'Uniform KL': 'KL'})
df_uniform['Dist'] = 'Uniform'
df_unigram = df[['Training Step', 'Model', 'Unigram KL']].rename(columns={'Unigram KL': 'KL'})
df_unigram['Dist'] = 'Unigram'
df_out = pd.concat([df_uniform, df_unigram])

plt.figure(figsize=(10, 6))
sns.lineplot(data=df_out, x='Training Step', y='KL', hue='Model',style='Dist', markers=False)
plt.xscale('log')
plt.savefig(f'../working_dir/{sys.argv[1]}/output/deduped_uniform.png')
plt.close()
