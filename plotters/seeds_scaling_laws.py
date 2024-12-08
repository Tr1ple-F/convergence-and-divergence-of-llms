import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import deduped_config, pos_tags, tagged_tokens, strip

df = pd.read_csv(f'../working_dir/{sys.argv[1]}/output/seeds_surprisal_dataframe.csv')

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
    sns.lineplot(data=frame, x='Revision', y='Surprisal', hue="Model", errorbar='sd')

    plt.title('Cross Entropy by Revision')
    plt.yscale('log')
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig(f'../working_dir/{sys.argv[1]}/output/cross_entropy_std_seeds.png') # Unexplained error for
    plt.close()

plot_ci(df)
