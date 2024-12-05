import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import sys
import json

# Load the dataframe
df = pd.read_csv(f'../working_dir/{sys.argv[1]}/output/average_dataframe.csv')

from utils import deduped_config, strip

config = deduped_config()
revisions = config['revisions']
models = config['model_names']

for x in models:
    df_plot = df[df['Model 1'] == strip(x)]
    df_plot = df_plot[df_plot['Revision 1'] == df_plot['Revision 2']]
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_plot, x='Revision 1', y='KL Average', hue='Model 2', markers=True)
    plt.xscale('log')
    plt.savefig(f'../working_dir/{sys.argv[1]}/output/deduped_no_seeds_{strip(x)}.png')
    plt.close()

print(df)

for x in revisions:
    print(strip(x))
    df_plot = df[df['Revision 1'] == int(strip(x))]
    print(df_plot)
    df_plot = df_plot[df_plot['Model 1'] == df_plot['Model 2']]
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_plot, x='Model 1', y='KL Average', hue='Revision 2', markers=True)
    # plt.xscale('log')
    plt.savefig(f'../working_dir/{sys.argv[1]}/output/deduped_no_seeds_{strip(x)}.png')
    plt.close()
