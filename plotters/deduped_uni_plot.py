import matplotlib.pyplot as plt
import pandas as pd
import sys
import seaborn as sns

df = pd.read_csv(f'../working_dir/{sys.argv[1]}/output/dist_average_dataframe.csv')
df = df[df['Model 1'] != '12b']
df = df[df['Model 1'] != '6.9b']

df_uniform = df[['Revision 1', 'Model 1', 'Uniform KL']].rename(columns={'Uniform KL': 'KL'})
df_uniform['Dist'] = 'Uniform'
df_unigram = df[['Revision 1', 'Model 1', 'Unigram KL']].rename(columns={'Unigram KL': 'KL'})
df_unigram['Dist'] = 'Unigram'
df_out = pd.concat([df_uniform, df_unigram])

plt.figure(figsize=(10, 6))
sns.lineplot(data=df_out, x='Revision 1', y='KL', hue='Model 1',style='Dist', markers=False)
plt.xscale('log')
plt.savefig(f'../working_dir/{sys.argv[1]}/output/deduped_uniform.png')
plt.close()
