import matplotlib.pyplot as plt
import pandas as pd
import sys
import seaborn as sns
from utils import pos_tags

df = pd.read_csv(f'../working_dir/{sys.argv[1]}/output/average_dataframe_seeds.csv')

# df = df[df['Revision 1'] != 143000]
# df = df[df['Revision 1'] != 128000]
# df = df[df['Revision 2'] != 143000]
# df = df[df['Revision 2'] != 128000]

# Rename Surprisal Average to Cross Entropy
df = df.rename(columns={'Surprisal Average': 'Cross Entropy'})
df = df.rename(columns={'Surprisal Average 2': 'Cross Entropy for Model 2'})

if sys.argv[2] == 'pos':
    for i, pos_tag in enumerate(pos_tags()):
        tag = pos_tag['tag']
        sns.scatterplot(data=df, x=f"KL Average - {tag}", y=f"Surprisal Average - {tag}")
        plt.title("Scatter Plot of KL Average vs Surprisal Average for " + tag)
        plt.savefig(f'../working_dir/{sys.argv[1]}/output/scatterplot_surprisal_kl_seeds_{i}.png')
        plt.close()

df = df[df['Seed 1'] != df['Seed 2']]

sns.scatterplot(data=df, x="KL Average", y="Cross Entropy", hue="Cross Entropy for Model 2")
plt.title("KL Average vs Cross Entropy")
plt.savefig(f'../working_dir/{sys.argv[1]}/output/scatterplot_surprisal_kl_all_sp_avg2.png')
plt.close()

sns.scatterplot(data=df, x="KL Average", y="Cross Entropy", hue="Revision 2")
plt.title("KL Average vs Cross Entropy")
plt.savefig(f'../working_dir/{sys.argv[1]}/output/scatterplot_surprisal_kl_all_rv2.png')
plt.close()

sns.scatterplot(data=df, x="KL Average", y="Cross Entropy", hue="Revision 1")
plt.title("KL Average vs Cross Entropy")
plt.savefig(f'../working_dir/{sys.argv[1]}/output/scatterplot_surprisal_kl_all_rv1.png')
plt.close()

sns.scatterplot(data=df, x="KL Average", y="Cross Entropy", hue="Model 1")
plt.title("KL Average vs Cross Entropy")
plt.savefig(f'../working_dir/{sys.argv[1]}/output/scatterplot_surprisal_kl_all_m1.png')
plt.close()

sns.scatterplot(data=df, x="KL Average", y="Cross Entropy", hue="Model 2")
plt.title("KL Average vs Cross Entropy")
plt.savefig(f'../working_dir/{sys.argv[1]}/output/scatterplot_surprisal_kl_all_m2.png')
plt.close()
