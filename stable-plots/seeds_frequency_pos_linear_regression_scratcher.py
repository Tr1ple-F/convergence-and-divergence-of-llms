import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import seaborn as sns
import sys
import matplotlib.pyplot as plt

def test1():
    df = pd.read_csv(f'../working_dir/{sys.argv[1]}/output/seeds_frequency_dataframe.csv')
    df = df[df['Seed 1'] != df['Seed 2']]
    df = df[df['Model 1'] == df['Model 2']]
    df = df[df['Revision 1'] == df['Revision 2']]
    # Average by model revision
    # Plot the evolution of the model over the rows
    df_grouped = df.groupby(['Model 1', 'Revision 1']).agg({'KL': 'mean'}).reset_index()
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_grouped, x='Revision 1', y='KL', hue='Model 1', marker='o')

    plt.xlabel('Revision 1')
    plt.ylabel('Average KL')
    plt.title('Average KL by Revision and Model')
    plt.xscale('log')
    plt.legend(title='Model 1')
    plt.savefig(f'../working_dir/{sys.argv[1]}/output/seeds_average_kl_by_revision_and_model.png')
    plt.close()

df = pd.read_csv(f'../working_dir/{sys.argv[1]}/output/seeds_frequency_pos_linear_regression_transpose.csv')

def plot_lg_by(list, appendix):
    for x in list:
        plt.plot(df.index, df[x], label=x)

    plt.xlabel('Row Index')
    plt.ylabel('Value')
    plt.title(f'{appendix} Evolution Over Rows')
    plt.legend()
    plt.savefig(f'../working_dir/{sys.argv[1]}/output/seeds_linear_regression_by_{appendix}.png')
    plt.close()

models = ['Model 1_14m', 'Model 1_160m', 'Model 1_31m', 'Model 1_410m', 'Model 1_70m']
frequency = ['Frequency']
pos_context_nouns = [f'POS Context_{x}' for x in ['NN', 'NNS', 'NNP', 'NNPS']]
pos_context_verbs = [f'POS Context_{x}' for x in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']]
pos_context_adjectives = [f'POS Context_{x}' for x in ['JJ', 'JJR', 'JJS']]
pos_context_adverbs = [f'POS Context_{x}' for x in ['RB', 'RBR', 'RBS']]
pos_context_other = [f'POS Context_{x}' for x in ['CD','DT', 'EX','IN', 'MD', 'PRP', 'PRP$', 'TO']]
pos_context_w = [f'POS Context_{x}' for x in ['WDT', 'WP', 'WRB']]
pos_nouns = [f'POS_{x}' for x in ['NN', 'NNS', 'NNP', 'NNPS']]
pos_verbs = [f'POS_{x}' for x in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']]
pos_adjectives = [f'POS_{x}' for x in ['JJ', 'JJR', 'JJS']]
pos_adverbs = [f'POS_{x}' for x in ['RB', 'RBR', 'RBS']]
pos_other = [f'POS_{x}' for x in ['CD','DT', 'EX', 'IN', 'MD', 'PRP', 'PRP$', 'TO']]
pos_w = [f'POS_{x}' for x in ['WDT', 'WP', 'WRB']]

plot_lg_by(models, 'Model')
plot_lg_by(frequency, 'Frequency')
plot_lg_by(pos_context_nouns, 'POS Context Nouns')
plot_lg_by(pos_context_verbs, 'POS Context Verbs')
plot_lg_by(pos_context_adjectives, 'POS Context Adjectives')
plot_lg_by(pos_context_adverbs, 'POS Context Adverbs')
plot_lg_by(pos_context_other, 'POS Context Other')
plot_lg_by(pos_context_w, 'POS Context W')
plot_lg_by(pos_nouns, 'POS Nouns')
plot_lg_by(pos_verbs, 'POS Verbs')
plot_lg_by(pos_adjectives, 'POS Adjectives')
plot_lg_by(pos_adverbs, 'POS Adverbs')
plot_lg_by(pos_other, 'POS Other')
plot_lg_by(pos_w, 'POS W')
