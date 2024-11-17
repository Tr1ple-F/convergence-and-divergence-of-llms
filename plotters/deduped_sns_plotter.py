import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from utils import pos_tags

df = pd.read_csv('../graphics/average_dataframe.csv')

for i, pos_tag in enumerate(pos_tags()):
    tag = pos_tag['tag']
    sns.scatterplot(data=df, x=f"KL Average - {tag}", y=f"Surprisal Average - {tag}")
    plt.title("Scatter Plot of KL Average vs Surprisal Average for " + tag)
    plt.savefig(f'../graphics/scatterplot_surprisal_kl_{i}.png')
    plt.close()

sns.scatterplot(data=df, x="KL Average", y="Surprisal Average")
plt.title("Scatter Plot of KL Average vs Surprisal Average for all")
plt.savefig('../graphics/scatterplot_surprisal_kl_all.png')
plt.close()
