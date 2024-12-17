import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys

from utils import tagged_tokens


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

nns_ids_1_before = [idx-1 for idx, tag in enumerate(tagged_tokens()) if tag[1] == 'NNS']
nns_ids = [idx for idx, tag in enumerate(tagged_tokens()) if tag[1] == 'NNS']
nns_ids_1_after = [idx+1 for idx, tag in enumerate(tagged_tokens()) if tag[1] == 'NNS']

# Load the dataframe
df = pd.read_csv(f'../working_dir/{sys.argv[1]}/output/seeds_frequency_dataframe.csv')

kl_data = np.load(f'../working_dir/{sys.argv[1]}/results/seeds/EleutherAI-pythia-410m-step143000-seed9-kl.npy')
temp = np.mean(kl_data[:, nns_ids], axis = 1)
print(temp[-2])
print(pd.DataFrame(kl_data[-2, nns_ids_1_before]))
print(pd.DataFrame(kl_data[-2, nns_ids]))
print(pd.DataFrame(kl_data[-2, nns_ids_1_after]))
# print(temp[-1])

# Build alternative frame
alt = df[df['Model 1'] == '410m']
alt = alt[alt['Model 2'] == '410m']
alt = alt[alt['Revision 1'] == 143000]
alt = alt[alt['Revision 2'] == 143000]
alt = alt[alt['Seed 1'] == 9]
alt = alt[alt['Seed 2'] == 7]
alt = alt[alt['POS'] == 'NNS']
avg_for_NNS = alt.groupby(['Model 1'])['KL'].mean().reset_index()
print(alt[['Token ID', 'KL']])
print(avg_for_NNS)
print("Loaded dataframe")
df['Binned Freq'] = df['Frequency'].apply(bin_freq_10)
print("Binned")
df_average = df.groupby(['Token ID', 'Model 1', 'Revision 1', 'Seed 1', 'Model 2', 'Revision 2', 'Seed 2', 'Binned Freq', 'POS']).mean().reset_index()
print(df_average)

# Can you generate a y where frequency is how often each binned freq occurs in the dataframe
y = df.groupby(['POS', 'Binned Freq']).count().reset_index()
y = y[y['Frequency'] > 50]
print(y)
plt.figure(figsize=(10, 6))
sns.barplot(data=y, x='POS', y='Frequency', hue='Binned Freq')
plt.xlabel('Frequency Bin')
plt.ylabel('Occurrence')
plt.savefig(f'../working_dir/{sys.argv[1]}/output/frequency_and_pos_bar_plot.png')

print("Bar plot done")
def plot(pos_tag):
    df_plot = df_average[df_average['POS'] == pos_tag]
    df_plot = df_plot[df_plot['Model 2'] == df_plot['Model 1']]
    df_plot = df_plot[df_plot['Model 1'] == '410m']
    df_plot = df_plot[df_plot['Revision 1'] == df_plot['Revision 2']]
    df_plot = df_plot[df_plot['Seed 1'] != df_plot['Seed 2']]
    df_plot = df_plot.merge(y[['POS', 'Binned Freq', 'Frequency']], on=['POS', 'Binned Freq'])
    print(df_plot)
    # Calculate average KL by revision
    avg_kl_by_revision = df_plot.groupby(['Revision 1', 'Binned Freq'])['KL'].mean().reset_index()
    print(avg_kl_by_revision)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_plot, x='Revision 1', y='KL', hue='Binned Freq')
    plt.xscale('log')
    plt.savefig(f'../working_dir/{sys.argv[1]}/output/frequency_and_pos_{pos_tag}.png')
    plt.close()

plot("NNS")

if False:
    plot("NN")
    plot("MD")
    plot("NNP")
    plot("DT")
    plot("PRP$")
    plot("PRP")
    plot("IN")
    plot("TO")
