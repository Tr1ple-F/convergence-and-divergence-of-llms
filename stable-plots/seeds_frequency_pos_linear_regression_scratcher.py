import pandas as pd
import seaborn as sns
import sys
import matplotlib.pyplot as plt

def test1():
    df = pd.read_csv(f'../working_dir/{sys.argv[1]}/output/seeds_frequency_dataframe.csv')
    df = df[df['Seed 1'] != df['Seed 2']]
    df = df[df['Model'] == df['Model 2']]
    df = df[df['Training Step'] == df['Training Step 2']]

    # Average by model revision
    df_grouped = df.groupby(['Model', 'Training Step']).agg({'KL': 'mean'}).reset_index()

    plt.figure(figsize=(10, 6))
    sns.set_context("notebook", font_scale=1.5)
    sns.set_style("whitegrid")
    palette = sns.color_palette("Set2")

    sns.lineplot(
        data=df_grouped,
        x='Training Step',
        y='KL',
        hue='Model',
        style='Model',
        markers=True,
        palette=palette
    )

    plt.xscale('log')
    plt.xlabel('Training Step')
    plt.ylabel(r'Expected convergence ($\mathbb{E}[\mathrm{conv}]$)')

    for xpos in [16, 256, 2000]:
        plt.axvline(x=xpos, color='gray', linestyle='--', linewidth=1.5)

    plt.legend(title='Model', loc='upper left', bbox_to_anchor=(0, 1))
    plt.tight_layout()
    plt.savefig(f'../working_dir/{sys.argv[1]}/output/seeds_average_kl_by_revision_and_model.png', bbox_inches='tight')
    plt.close()

# Load second DataFrame for the second plot
df = pd.read_csv(f'../working_dir/{sys.argv[1]}/output/seeds_frequency_pos_linear_regression_transpose.csv')

# Rename model columns
models = ['Model 1_14m', 'Model 1_31m', 'Model 1_70m', 'Model 1_160m', 'Model 1_410m']
models_short = [x.replace('Model 1_', '') for x in models]
rename_dict = {models[i]: models_short[i] for i in range(len(models))}
df.rename(columns=rename_dict, inplace=True)

def plot_lg_by(columns, appendix, symbol="α"):
    plt.figure(figsize=(10, 6))
    sns.set_context("notebook", font_scale=1.5)
    sns.set_style("whitegrid")
    palette = sns.color_palette("Set2")

    for idx, col in enumerate(columns):
        plt.plot(df['Revision'], df[col], label=f'{col}', color=palette[idx % len(palette)], linestyle='-', marker='o')

    plt.xscale('log')
    plt.xlabel('Training Step')
    plt.ylabel(fr'Fitted {symbol} Value')

    for xpos in [16, 256, 2000]:
        plt.axvline(x=xpos, color='gray', linestyle='--', linewidth=1.5)

    plt.legend(title=appendix, loc='upper left', bbox_to_anchor=(0, 1))
    plt.tight_layout()
    plt.savefig(f'../working_dir/{sys.argv[1]}/output/seeds_linear_regression_by_{appendix}.png', bbox_inches='tight')
    plt.close()

# Example usage would be like:
# plot_lg_by(['14m', '31m', '70m', '160m', '410m'], 'Model Size')

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

# Symbol delta
plot_lg_by(models_short, 'Model', 'δ')
plot_lg_by(frequency, 'Frequency', 'α')
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
