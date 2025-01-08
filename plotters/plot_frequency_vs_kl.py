import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import sys

def bin_freq_10(x):
    if x < 10:
        return '0-10'
    elif x < 100:
        return '10-100'
    elif x < 1000:
        return '10^3-10^4'
    elif x < 10000:
        return '10^4-10^5'
    elif x < 100000:
        return '10^5-10^6'
    else:
        return '10^6+'

def bin_freq_5(x):
    if x < 10:
        return '0-10'
    elif x < 100:
        return '10-100'
    elif x < 1000:
        return '100-1000'
    elif x < 5000:
        return '1000-5000'
    elif x < 10000:
        return '5000-10000'
    elif x < 50000:
        return '10000-50000'
    elif x < 100000:
        return '50000-100000'
    elif x < 500000:
        return '100000-500000'
    else:
        return '500000+'

def bin_freq_10_steps(x):
    import math
    if x <= 0:
        return '0'
    log_bin = int(math.log10(x))
    step = int((x / (10 ** log_bin)) * 10)
    return f'{10 ** log_bin}-{10 ** (log_bin + 1)}: step {step}'


# Load the dataframe
df = pd.read_csv(f'../working_dir/{sys.argv[1]}/output/seeds_frequency_dataframe.csv')
df = df.drop(columns=["POS","POS Context"])
print("Loaded dataframe")
df['Binned Freq'] = df['Frequency'].apply(bin_freq_10)
print("Binned")
df_average = df.groupby(['Model', 'Training Step', 'Seed 1', 'Model 2', 'Training Step 2', 'Seed 2', 'Binned Freq']).mean().reset_index()
print("Grouped")



df_plot = df_average[df_average['Model'].isin(["14m", "410m"])]
df_plot = df_plot[df_plot['Model 2'] == df_plot['Model']]
df_plot = df_plot[df_plot['Training Step'] == df_plot['Training Step 2']]
df_plot = df_plot[df_plot['Seed 1'] != df_plot['Seed 2']]
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_plot, x='Training Step', y='KL', hue='Binned Freq', style="Model")
plt.xscale('log')
plt.savefig(f'../working_dir/{sys.argv[1]}/output/frequency_result.png')
plt.close()

exit()

for x in ['14m', '31m', '70m', '160m', '410m']:
    df_plot = df_average[df_average['Model'] == x]
    df_plot = df_plot[df_plot['Model 2'] == x]
    df_plot = df_plot[df_plot['Training Step'] == df_plot['Training Step 2']]
    df_plot = df_plot[df_plot['Seed 1'] != df_plot['Seed 2']]
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_plot, x='Training Step', y='KL', hue='Binned Freq', markers=True)
    plt.xscale('log')
    plt.savefig(f'../working_dir/{sys.argv[1]}/output/frequency_test_{x}.png')
    plt.close()