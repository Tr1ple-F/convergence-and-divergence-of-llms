import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import sys

def bin_freq(x):
    if x < 10:
        return '0-10'
    elif x < 100:
        return '10-100'
    elif x < 1000:
        return '100-1000'
    elif x < 10000:
        return '1000-10000'
    elif x < 100000:
        return '10000-100000'
    else:
        return '100000+'

# Load the dataframe
df = pd.read_csv(f'../working_dir/{sys.argv[1]}/output/seeds_frequency_dataframe.csv')

df['Binned Freq'] = df['Frequency'].apply(bin_freq)
df_average = df.groupby(['Model 1', 'Revision 1', 'Seed 1', 'Model 2', 'Revision 2', 'Seed 2', 'Binned Freq']).mean().reset_index()

for x in ['14m', '31m', '70m', '160m', '410m']:
    df_plot = df_average[df_average['Model 1'] == x]
    df_plot = df_plot[df_plot['Model 2'] == x]
    df_plot = df_plot[df_plot['Revision 1'] == df_plot['Revision 2']]
    df_plot = df_plot[df_plot['Seed 1'] != df_plot['Seed 2']]
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_plot, x='Revision 1', y='KL', hue='Binned Freq', markers=True)
    plt.xscale('log')
    plt.savefig(f'../working_dir/{sys.argv[1]}/output/frequency_test_{x}.png')
    plt.close()
