import matplotlib.pyplot as plt
import pandas as pd
import sys
import seaborn as sns

df = pd.read_csv(f'../working_dir/{sys.argv[1]}/output/frequency_dataframe.csv')

sns.scatterplot(data=df, x=f"Frequency", y=f"KL")
plt.title("Frequency vs KL")
plt.savefig(f'../working_dir/{sys.argv[1]}/output/freq_vs_kl.png')
plt.close()
