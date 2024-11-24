import matplotlib.pyplot as plt
import pandas as pd
import sys
import seaborn as sns

df = pd.read_csv(f'../working_dir/{sys.argv[1]}/output/frequency_dataframe_small.csv')
df = df[df["Frequency"] < 200000]

sns.scatterplot(data=df, x="Frequency", y="KL", hue="Model 2")
plt.title("Frequency vs KL")
plt.savefig(f'../working_dir/{sys.argv[1]}/output/freq_vs_kl_small.png')
plt.close()
