import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import sys
import json

# Load the dataframe
df = pd.read_csv(f'../working_dir/{sys.argv[1]}/output/deduped_pos_current_dataframe.csv')

from utils import deduped_config, strip

config = deduped_config()
revisions = config['revisions']
models = config['model_names']

for x in models:
    df_plot = df[df['Model'] == strip(x)]
    df_plot = df_plot[df_plot['Model 2'] != "12b"]
    df_plot = df_plot[df_plot['Model 2'] != "6.9b"]
    df_plot = df_plot[df_plot['Model 2'] != df_plot['Model']]
    df_plot = df_plot[df_plot['Training Step'] == df_plot['Training Step 2']]
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_plot, x='Training Step', y='KL Average', hue='Model 2', markers=True)
    plt.xscale('log')
    plt.legend(prop={'size': 16})
    plt.savefig(f'../working_dir/{sys.argv[1]}/output/deduped_model_{strip(x)}_to_others.png')
    plt.close()

exit()

for x in revisions:
    df_plot = df[df['Training Step'] == int(strip(x))]
    df_plot = df_plot[df_plot['Model'] == df_plot['Model 2']]
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_plot, x='Model', y='KL Average', hue='Revision 2', markers=True)
    # plt.xscale('log')
    plt.savefig(f'../working_dir/{sys.argv[1]}/output/deduped_no_seeds_{strip(x)}.png')
    plt.close()
