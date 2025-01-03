import matplotlib.pyplot as plt
import pandas as pd
import sys
import seaborn as sns

df = pd.read_csv(f'../working_dir/{sys.argv[1]}/output/deduped_pos_{sys.argv[2]}_dataframe.csv')
df = df[df['Model 1'] != '70m']
df = df[df['Model 1'] != '160m']
df = df[df['Model 2'] != '70m']
df = df[df['Model 2'] != '160m']

df = df.rename(columns={'Surprisal Average': 'Cross Entropy'})
df = df.rename(columns={'Surprisal Average 2': 'Cross Entropy for Model 2'})
df = df[df['Model 1'] != df['Model 2']]
df["Cross-Entropy Difference"] = df["Cross Entropy"] - df["Cross Entropy for Model 2"]

sns.scatterplot(data=df, x="KL Average", y="Cross-Entropy Difference", hue="Model 1")
plt.title("KL Average vs Cross Entropy Difference")
plt.savefig(f'../working_dir/{sys.argv[1]}/output/scatterplot_cross_entropy_difference_{sys.argv[2]}.png')
plt.close()
