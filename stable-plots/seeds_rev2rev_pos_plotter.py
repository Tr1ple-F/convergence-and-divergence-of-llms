import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import sys
import json

pos_index = sys.argv[3]

df = pd.read_csv(f'../working_dir/{sys.argv[1]}/output/seeds_pos_{pos_index}_dataframe.csv')

with open(f'../common/pos_tags.json', 'r') as f:
    pos_tags = json.load(f)

def plot_internal(value_vars, appendix=""):
    for x in ['14m', '31m', '70m', '160m', '410m']:
        df_plot = df.melt(id_vars=['Model', 'Model 2', 'Training Step', 'Training Step 2', 'Seed 1', 'Seed 2'], value_vars=value_vars, var_name='PoS Tag', value_name='Average KL')
        df_plot = df_plot[df_plot['Model'] == x]
        df_plot = df_plot[df_plot['Model 2'] == x]
        df_plot = df_plot[df_plot['Training Step'] == df_plot['Training Step 2']]
        df_plot = df_plot[df_plot['Seed 1'] != df_plot['Seed 2']]
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_plot, x='Training Step', y='Average KL', hue='PoS Tag', errorbar='ci', markers=True)
        plt.xscale('log')
        plt.savefig(f'../working_dir/{sys.argv[1]}/output/seeds_rev2rev_model_{x}{appendix}_{pos_index}.png')
        plt.close()

def plot_no_filters():
    df_plot = df.copy()
    df_plot = df_plot[df_plot['Model'] == df_plot['Model 2']]
    df_plot = df_plot[df_plot['Training Step'] == df_plot['Training Step 2']]
    df_plot = df_plot[df_plot['Seed 1'] != df_plot['Seed 2']]
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_plot, x='Training Step', y='KL Average', hue='Model', errorbar='ci', markers=True)
    plt.xscale('log')
    plt.savefig(f'../working_dir/{sys.argv[1]}/output/seeds_no_filters.png')
    plt.close()

plot_no_filters()
noun_vars = [f'KL Average - {x}' for x in ["NN", "NNS", "NNP"]]

if sys.argv[2] == "other":
    plot_internal([f'KL Average - {x}' for x in ["PRP$", "DT", "CC", "JJ", "IN", "TO", "RB", "MD","PRP"]], "_other")

if sys.argv[2] == "nouns":
    plot_internal(noun_vars, "_nouns")

if sys.argv[2] == "verbs":
    plot_internal([f'KL Average - {x}' for x in ["VB", "VBG", "VBN", "VBD", "VBZ"]], "_verbs")

if sys.argv[2] == "grouped":
    plot_internal([f'KL Average - {x}' for x in ["JJ", "IN", "DT", 'MD', 'PRP', "Nouns", "Verbs"]], "_grouped")

if sys.argv[2] == "all":
    plot_internal([f'KL Average - {x}' for x in ["PRP$", "DT", "CC", "JJ", "IN", "TO", "RB", "MD","PRP"]], "_other")
    plot_internal(noun_vars, "_nouns")
    plot_internal([f'KL Average - {x}' for x in ["VB", "VBG", "VBN", "VBD", "VBZ"]], "_verbs")
    plot_internal([f'KL Average - {x}' for x in ["JJ", "IN", "DT", 'MD', 'PRP', "Nouns", "Verbs"]], "_grouped")
