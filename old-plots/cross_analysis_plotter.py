import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(f'../working_dir/{sys.argv[1]}/output/cross_dataframe.csv')

plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='Revision', y='KL Average', hue='Model', errorbar="sd")
plt.xscale('log')
plt.savefig(f'../working_dir/{sys.argv[1]}/output/cross_analysis.png')
plt.close()
