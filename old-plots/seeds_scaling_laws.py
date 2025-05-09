import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(f'../working_dir/{sys.argv[1]}/output/seeds_surprisal_dataframe.csv')

def plot_individual(frame):
    frame = frame[frame['Model'] == sys.argv[2]]

    plt.figure(figsize=(10, 6))
    sns.set_context("notebook", font_scale=1.5)
    sns.set_style("whitegrid")
    palette = sns.color_palette("Set2")

    sns.lineplot(
        data=frame,
        x='Revision',
        y='Surprisal',
        color=palette[0]  # Single color since no hue
    )

    plt.xscale('log')
    plt.xlabel('Revision')
    plt.ylabel(r'Expected convergence ($\mathbb{E}[\mathrm{conv}]$)')

    # Add vertical lines
    for xpos in [16, 256, 2000]:
        plt.axvline(x=xpos, color='gray', linestyle='--', linewidth=1.5)

    plt.tight_layout()
    plt.savefig(f'../working_dir/{sys.argv[1]}/output/cross_entropy_{sys.argv[2]}.png', bbox_inches='tight')
    plt.close()

def plot_ci(frame):
    plt.figure(figsize=(10, 6))
    sns.set_context("notebook", font_scale=1.5)
    sns.set_style("whitegrid")
    palette = sns.color_palette("Set2")

    sns.lineplot(
        data=frame,
        x='Revision',
        y='Surprisal',
        hue='Model',
        style='Model',
        markers=True,
        errorbar='sd',
        palette=palette
    )

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Revision')
    plt.ylabel(r'Expected convergence ($\mathbb{E}[\mathrm{conv}]$)')

    # Add vertical lines
    for xpos in [16, 256, 2000]:
        plt.axvline(x=xpos, color='gray', linestyle='--', linewidth=1.5)

    plt.legend(loc='upper left', bbox_to_anchor=(0, 1))
    plt.tight_layout()
    plt.savefig(f'../working_dir/{sys.argv[1]}/output/cross_entropy_std_seeds.png', bbox_inches='tight')
    plt.close()

# Call the plotting functions
plot_ci(df)
plot_individual(df)
