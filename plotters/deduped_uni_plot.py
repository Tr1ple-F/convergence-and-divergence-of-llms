import matplotlib.pyplot as plt
import pandas as pd
import sys
import seaborn as sns

df = pd.read_csv(f'../working_dir/{sys.argv[1]}/output/dist_average_dataframe.csv')

plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='Revision 1', y='Uniform KL', hue='Model 1')
plt.xscale('log')
plt.savefig(f'../working_dir/{sys.argv[1]}/output/deduped_uniform.png')
plt.close()

plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='Revision 1', y='Unigram KL', hue='Model 1')
plt.xscale('log')
plt.savefig(f'../working_dir/{sys.argv[1]}/output/deduped_unigram.png')
plt.close()
