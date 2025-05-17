import json
import sys

import ipdb
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.family'] = "serif"

def deduped_config():
    with open(f'../working_dir/{sys.argv[1]}/deduped_config.json') as f:
        return json.load(f)

def seeds_config():
    with open(f'../working_dir/{sys.argv[1]}/seeds_config.json') as f:
        return json.load(f)

def tagged_tokens():
    with open(f'../working_dir/{sys.argv[1]}/pos_tagged_tokens.json', 'r') as f:
        return json.load(f)

def pos_tags():
    with open('../common/pos_tags.json', 'r') as f:
        return json.load(f)

def strip(name):
    if "EleutherAI" in name:
        return name.replace("EleutherAI/pythia-", "").replace("-deduped", "")
    if "step" in name:
        return name.replace("step", "")
    if "KL Average - " in name:
        return name.replace('KL Average - ', '')
    if "Surprisal Average - " in name:
        return name.replace('Surprisal Average - ', '')
    return name

def styled_plot(
        df_plot,
        x_name,
        y_name,
        hue,
        style,
        x_label,
        y_label,
        save_loc,
        y_log = False,
        order_legend = True,
        y_scale = 6,
        legend_include = None
):
    plt.figure(figsize=(10, y_scale))
    palette = sns.color_palette("Set2")
    sns.set_theme("notebook", "whitegrid", palette=palette, font="serif", font_scale=1.75)

    sns.lineplot(
        data=df_plot,
        x=x_name,
        y=y_name,
        hue=hue,
        style=style,
        markers=True,
        markersize=10,  # Increased marker size
        linewidth=2,
        errorbar='sd',
        palette=palette
    )

    plt.xscale('log')
    if y_log:
        plt.yscale('log')
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Add vertical lines
    for xpos in [16, 256, 2000]:
        plt.axvline(x=xpos, color='gray', linestyle='--', linewidth=2)

    if order_legend:
        max_x = df_plot[x_name].max()
        df_last = df_plot[df_plot[x_name] == max_x].copy()
        df_last = df_last.groupby(hue)[y_name].mean().sort_values(ascending=False)
        order = df_last.index.tolist()

        handles, labels = plt.gca().get_legend_handles_labels()
        label_to_handle = dict(zip(labels, handles))

        new_labels = [f"{strip(label)}" for label in labels if label != '']
        label_map = dict(zip(labels, new_labels))

        # Filter and order handles/labels based on sorted order
        ordered_handles = [label_to_handle[label] for label in order]
        ordered_clean_labels = [label_map[label] for label in order]
    else:
        ordered_handles, labels = plt.gca().get_legend_handles_labels()
        ordered_clean_labels = [f"{strip(label)}" for label in labels if label != '']

    if legend_include:
        from_a, to_b = legend_include
        ipdb.set_trace()
        ordered_handles = ordered_handles[from_a:to_b]
        ordered_clean_labels = ordered_clean_labels[from_a:to_b]

    plt.legend(handles=ordered_handles, labels=ordered_clean_labels, loc='upper left', bbox_to_anchor=(0, 1))

    plt.tight_layout()
    plt.savefig(save_loc, bbox_inches='tight')
    plt.close()