import sys
import pandas as pd
from utils import styled_plot

pos_index = sys.argv[3]

df = pd.read_csv(f'../working_dir/{sys.argv[1]}/output/seeds_surprisal_dataframe.csv')
df2 = pd.read_csv(f'../working_dir/{sys.argv[1]}/output/seeds_pos_{pos_index}_dataframe.csv')

path = f'../working_dir/{sys.argv[1]}/output/cross_entropy_std_seeds.png'
y_label = r'Cross Entropy ($H$)'
styled_plot(df, 'Revision', 'Surprisal', 'Model', 'Model', 'Training Step', y_label, path, y_log=True)

def plot_internal(value_vars, appendix=""):
    for model_size in ['410m']: # ['14m', '31m', '70m', '160m', '410m']:
        df_plot = df2.melt(
            id_vars=['Model', 'Model 2', 'Training Step', 'Training Step 2', 'Seed 1', 'Seed 2'],
            value_vars=value_vars,
            var_name='PoS Tag',
            value_name='Average KL'
        )
        df_plot = df_plot[
            (df_plot['Model'] == model_size) &
            (df_plot['Model 2'] == model_size) &
            (df_plot['Training Step'] == df_plot['Training Step 2']) &
            (df_plot['Seed 1'] != df_plot['Seed 2'])
            ]

        y_label = r'Cross Entropy ($H$)'
        save_loc = f'../working_dir/{sys.argv[1]}/output/seeds_rev2rev_model_{model_size}{appendix}_{pos_index}.png'

        styled_plot(df_plot, 'Training Step', 'Average KL', 'PoS Tag', 'PoS Tag', 'Training Step', y_label, save_loc, y_log=True)

# ====== Run the functions based on sys.argv[2] ======

noun_vars = [f'Surprisal Average - {x}' for x in ["NN", "NNS", "NNP"]]
other_vars = [f'Surprisal Average - {x}' for x in ["PRP$", "DT", "CC", "JJ", "IN", "TO", "RB", "MD", "PRP"]]
verb_vars = [f'Surprisal Average - {x}' for x in ["VB", "VBG", "VBN", "VBD", "VBZ"]]
grouped_vars = [f'Surprisal Average - {x}' for x in ["JJ", "IN", "DT", "MD", "PRP", "Nouns", "Verbs"]]

if sys.argv[2] == "other":
    plot_internal(other_vars,"_other")

if sys.argv[2] == "nouns":
    plot_internal(noun_vars, "_nouns")

if sys.argv[2] == "verbs":
    plot_internal(verb_vars, "_verbs")

if sys.argv[2] == "grouped":
    plot_internal(grouped_vars, "_grouped")

if sys.argv[2] == "all":
    plot_internal(other_vars,"_other")
    plot_internal(noun_vars, "_nouns")
    plot_internal(verb_vars, "_verbs")
    plot_internal(grouped_vars, "_grouped")