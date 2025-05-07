import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import sys
import json

# Load the dataframe
df = pd.read_csv(f'../working_dir/{sys.argv[1]}/output/average_dataframe_seeds_neg1.csv')

# Plot 0
with open(f'../common/pos_tags.json', 'r') as f:
    pos_tags = json.load(f)

def plot_internal(value_vars, appendix=""):
    for x in ['14m', '31m', '70m', '160m', '410m']:
        df_plot = df.melt(id_vars=['Model 1', 'Model 2', 'Revision 1', 'Revision 2', 'Seed 1', 'Seed 2'], value_vars=value_vars, var_name='PoS Tag', value_name='Average KL')
        df_plot = df_plot[df_plot['Model 1'] == x]
        df_plot = df_plot[df_plot['Model 2'] == x]
        df_plot = df_plot[df_plot['Revision 1'] == df_plot['Revision 2']]
        df_plot = df_plot[df_plot['Seed 1'] != df_plot['Seed 2']]
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_plot, x='Revision 1', y='Average KL', hue='PoS Tag', errorbar='sd', markers=True)
        plt.xscale('log')
        plt.savefig(f'../working_dir/{sys.argv[1]}/output/tiago_test_pos_tag_{x}{appendix}.png')
        plt.close()

# value_vars = [f'KL Average - {x}' for x in ["VB", "CC", "CD", "JJ", "PRP", "RP", "RB", "WP", "DT", "IN", "MD", "NN"]]
# value_vars = [f'KL Average - {x["tag"]}' for x in pos_tags]

noun_vars = [f'KL Average - {x}' for x in ["NN", "NNS", "NNP"]]

def noun_plot():
    df_plot = df.melt(id_vars=['Model 1', 'Model 2', 'Revision 1', 'Revision 2', 'Seed 1', 'Seed 2'], value_vars=noun_vars, var_name='PoS Tag', value_name='Average KL')
    df_plot = df_plot[df_plot['Model 1'] == df_plot['Model 2']]
    df_plot = df_plot[df_plot['Model 1'] != '31m']
    df_plot = df_plot[df_plot['Model 1'] != '160m']
    df_plot = df_plot[df_plot['Revision 1'] == df_plot['Revision 2']]
    df_plot = df_plot[df_plot['Seed 1'] != df_plot['Seed 2']]
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_plot, x='Revision 1', y='Average KL', hue='PoS Tag', style="Model 1", markers=False, errorbar='ci')
    plt.xscale('log')
    plt.savefig(f'../working_dir/{sys.argv[1]}/output/special_noun_plot_sd.png')
    plt.close()

if sys.argv[2] == "extra":
    noun_plot()

if sys.argv[2] == "other":
    plot_internal([f'KL Average - {x}' for x in ["PRP$", "DT", "CC", "JJ", "IN", "TO", "RB", "MD","PRP"]], "_other")

if sys.argv[2] == "nouns":
    plot_internal(noun_vars, "_nouns")

if sys.argv[2] == "verbs":
    plot_internal([f'KL Average - {x}' for x in ["VB", "VBG", "VBN", "VBD", "VBZ", "VBP"]], "_verbs")

if sys.argv[2] == "grouped":
    plot_internal([f'KL Average - {x}' for x in ["JJ", "IN", "DT", "Nouns", "Verbs"]], "_grouped")
