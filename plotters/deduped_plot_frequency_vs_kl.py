import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import sys
from utils import deduped_config, strip

revs = [int(strip(x)) for x in deduped_config()['revisions']]

list_of_bins = ['0-10', '10-100', '100-1000', '1000-10000', '10000-100000', '100000+']

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
df = pd.read_csv(f'../working_dir/{sys.argv[1]}/output/frequency_dataframe.csv')

print("Loaded dataframe")

df['Binned Freq'] = df['Frequency'].apply(bin_freq)
print
df_average = df.groupby(['Model 1', 'Revision 1', 'Model 2', 'Revision 2', 'Binned Freq']).mean().reset_index()

print(df_average)

frame2 = df_average[df_average['Model 1'] == df['Model 2']]
frame2 = frame2[frame2['Model 1'] != "12b"]
frame2 = frame2[frame2['Model 1'] != "6.9b"]

frame2 = frame2[frame2['Revision 1'] != 143000]
frame2 = frame2[frame2.apply(lambda row: row['Revision 2'] == revs[revs.index(row['Revision 1']) + 1], axis=1)]

def analyze(row):
    for bin in list_of_bins:
        if row['KL Average'] > 0.001:
            pass
            # if row[f'KL Average - {tag}'] > c1*row[f'KL Average']:
                # print(f"! Revision: {row['Revision 1']}, Model: {row['Model 1']} -> {row['Model 2']}, Tag: {tag}, %: {row[f'KL Average - {tag}']/row[f'KL Average'] * 100}")

for i, row in frame2.iterrows():
    analyze(row)
