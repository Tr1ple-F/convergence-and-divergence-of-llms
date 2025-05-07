import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import sys
import json

# Load the dataframe
df = pd.read_csv(f'../working_dir/{sys.argv[1]}/output/average_dataframe_seeds.csv')

# Plot 0
with open(f'../common/pos_tags.json', 'r') as f:
    pos_tags = json.load(f)

df_plot = df.copy()
df_plot = df_plot[df_plot['Model 1'] == df_plot['Model 2']]
df_plot = df_plot[df_plot['Revision 1'] == df_plot['Revision 2']]
df_plot = df_plot[df_plot['Seed 1'] != df_plot['Seed 2']]
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_plot, x='Revision 1', y='KL Average', hue='Model 1', errorbar='sd', markers=True)
plt.xscale('log')
plt.savefig(f'../working_dir/{sys.argv[1]}/output/plot_56_no_filters.png')
plt.close()
