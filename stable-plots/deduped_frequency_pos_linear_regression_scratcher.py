import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import seaborn as sns
import sys
import matplotlib.pyplot as plt

df = pd.read_csv(f'../working_dir/{sys.argv[1]}/output/deduped_frequency_pos_linear_regression_transpose.csv')

def plot_lg_by(list, appendix):
    for x in list:
        plt.plot(df.index, df[x], label=x)

    plt.xlabel('Row Index')
    plt.ylabel('Value')
    plt.title(f'{appendix} Evolution Over Rows')
    plt.legend()
    plt.savefig(f'../working_dir/{sys.argv[1]}/output/deduped_linear_regression_by_{appendix}.png')
    plt.close()

models = ['Model 1_70m', 'Model 1_160m', 'Model 1_410m', 'Model 1_1b', 'Model 1_1.4b', 'Model 1_2.8b', 'Model 1_6.9b', 'Model 1_12b']
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
