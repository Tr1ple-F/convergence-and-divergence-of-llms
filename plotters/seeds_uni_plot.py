import matplotlib.pyplot as plt
import pandas as pd
import sys
import seaborn as sns

df = pd.read_csv(f'../working_dir/{sys.argv[1]}/output/seeds_uni_dataframe.csv')

df_uniform = df[['Training Step', 'Model', 'Uniform KL']].rename(columns={'Uniform KL': 'KL'})
df_uniform['Distribution'] = 'Uniform'
df_unigram = df[['Training Step', 'Model', 'Unigram KL']].rename(columns={'Unigram KL': 'KL'})
df_unigram['Distribution'] = 'Unigram'
df_out = pd.concat([df_uniform, df_unigram])

plt.figure(figsize=(10, 6))
sns.set_context("notebook", font_scale=1.5)  # (i) Increase font sizes
sns.set_style("whitegrid")  # Improve background style
palette = sns.color_palette("Set2")  # (ii) Change color palette

sns.lineplot(
    data=df_out,
    x='Training Step',
    y='KL',
    hue='Model',
    style='Distribution',  # (iii) Use different line styles for distributions
    errorbar='sd',
    palette=palette
)

plt.xscale('log')
plt.xlabel('Training Step')
plt.ylabel(r'Expected convergence ($\mathbb{E}[\mathrm{conv}]$)')  # (iv) Update y-axis label with LaTeX

for xpos in [16, 256, 2000]:
    plt.axvline(x=xpos, color='gray', linestyle='--', linewidth=1.5)

plt.tight_layout()  # (v) Remove useless whitespace

# Save figure
plt.savefig(f'../working_dir/{sys.argv[1]}/output/seeds_against_dist.png', bbox_inches='tight')
plt.close()