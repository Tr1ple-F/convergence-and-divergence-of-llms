import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import deduped_config, pos_tags, tagged_tokens, strip

df = pd.read_csv(f'../working_dir/{sys.argv[1]}/output/average_dataframe.csv')

def plot_pos(pos, log = False, revision_cutoff = 500):
    # Create a bar plot
    df_melted = df[df['Revision 1'] < revision_cutoff]
    # df_melted = df_melted.melt(id_vars=['Model 1', 'Model 2', 'Revision 1', 'Revision 2'], var_name='Metric', value_vars=[f'Surprisal Average - {pos}'], value_name='Surprisal')

    sns.lineplot(data=df_melted, x='Revision 1', y=f'Surprisal Average - {pos}', hue='Model 1')
    plt.title('MD and PRP Surprisal Average by Revision')
    if log:
        plt.xscale('log')
    plt.tight_layout()
    plt.savefig(f'../working_dir/{sys.argv[1]}/output/{pos}_cross_entropy.png')
    plt.close()

def plot_pos2(pos):
    df_melted = df[df['Revision 1'] > 64000]
    sns.lineplot(data=df_melted, x='Revision 1', y=f'Surprisal Average - {pos}', hue='Model 1')
    plt.title('MD and PRP Surprisal Average by Revision')
    # plt.yscale('log')
    # plt.xscale('log')
    plt.tight_layout()
    plt.savefig(f'../working_dir/{sys.argv[1]}/output/{pos}_cross_entropy.png')
    plt.close()

#plot_pos("MD", log = False, revision_cutoff = 500)
#plot_pos("PRP", log = False, revision_cutoff = 500)
#plot_pos("TO", log = False, revision_cutoff = 500)
#plot_pos("NNS", log = False, revision_cutoff = 500)
plot_pos2("PRP$")
plot_pos2("NNP")
