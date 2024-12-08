import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(f'../working_dir/{sys.argv[1]}/output/begin_end_dataframe.csv')
df = df[df['Model 1']  == '2.8b']
df = df[df['Revision 1'] == df['Revision 2']]

plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='Revision 1', y='Begin KL Average', hue='Model 2')
plt.xscale('log')
plt.savefig(f'../working_dir/{sys.argv[1]}/output/be_analysis_begin.png')
plt.close()

plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='Revision 1', y='End KL Average', hue='Model 2')
plt.xscale('log')
plt.savefig(f'../working_dir/{sys.argv[1]}/output/be_analysis_end.png')
plt.close()

plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='Revision 1', y='KL Average', hue='Model 2')
plt.xscale('log')
plt.savefig(f'../working_dir/{sys.argv[1]}/output/be_analysis_total.png')
plt.close()
