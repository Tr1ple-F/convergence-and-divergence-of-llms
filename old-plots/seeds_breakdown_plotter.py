import matplotlib.pyplot as plt
import pandas as pd
import sys
import seaborn as sns

df = pd.read_csv(f'../working_dir/{sys.argv[1]}/output/average_dataframe_seeds.csv')
df = df[df["Revision 2"].isin([128000, 143000])]
df = df[df["Revision 1"].isin([128000, 143000])]
df = df[df["Model 1"].isin(["410m"])]
print(df)

tag = 'CC'
sns.scatterplot(data=df, x=f"KL Average - {tag}", y=f"Surprisal Average - {tag}", hue="Model 2")
plt.title("Scatter Plot of KL Average vs Surprisal Average for " + tag)
plt.savefig(f'../working_dir/{sys.argv[1]}/output/scatterplot_surprisal_kl_seeds_breakdown_2x100k_filter_rev1.png')
plt.close()
