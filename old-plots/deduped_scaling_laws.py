import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import deduped_config, pos_tags, tagged_tokens, strip

df = pd.read_csv(f'../working_dir/{sys.argv[1]}/output/surprisal_dataframe.csv')
df = df[df['Model'].isin(["70m", "160m", "410m", "1b", "1.4b", "2.8b"])]
df = df[df['Revision'] != 143000]
df = df[df['Revision'] != 128000]

def plot_individual(frame):
    frame = frame[frame['Model'] == sys.argv[2]]

    # value_vars = [f'KL Average - {x}' for x in ["VB", "CC", "CD", "JJ", "PRP", "RP", "RB", "WP", "DT", "IN", "MD", "NN"]]
    list = ["CD", "CC", "RP", "RB", "TO", "DT", "WDT", "WP", "MD", "PRP"]
    list2 = ["PRP$", "VB", "JJ", "IN", "NN", "NNS", "NNP", "NNPS", "VBG", "VBN", "VBD", "VBZ"]

    # sns.lineplot(data=df_melted2, x='Revision', y='Surprisal', hue='Metric')
    sns.lineplot(data=frame, x='Revision', y='Surprisal')

    plt.title('Cross Entropy by Revision and Metric')
    # plt.yscale('log')
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig(f'../working_dir/{sys.argv[1]}/output/cross_entropy_{sys.argv[2]}.png')
    plt.close()

def plot_ci(frame):
    sns.lineplot(data=frame, x='Revision', y='Cross Entropy', hue="PoS Tag", errorbar='ci')

    plt.title('Cross Entropy by Revision')
    plt.yscale('log')
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig(f'../working_dir/{sys.argv[1]}/output/cross_entropy_tags.png') # Unexplained error for
    plt.close()

def plot_std(frame):

    sns.lineplot(data=frame, x='Revision', y='Cross Entropy', hue="PoS Tag")

    plt.title('Standard Deviation of Cross Entropy by Revision and PoS Tag')
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig(f'../working_dir/{sys.argv[1]}/output/cross_entropy_std.png') # Unexplained error for
    plt.close()



value_vars = [f'Surprisal Average - {x}' for x in ["VB", "JJ", "IN", "NN", "NNS", "NNP", "NNPS", "VBG", "VBN", "VBD", "VBZ"]]
melted = df.melt(id_vars=['Model', 'Revision'], value_vars=value_vars, var_name='PoS Tag', value_name='Cross Entropy')
print(melted)
plot_ci(melted)

# For the melted data frame calculate the standard deviation of the cross entropy values per Revision
df2 = melted.drop(columns=['Model'])
print(df2)
std = df2.groupby(['Revision', 'PoS Tag']).std().reset_index()
print(std)

plot_std(std)
