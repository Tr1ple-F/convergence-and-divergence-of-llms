import matplotlib.pyplot as plt
import numpy as np

from utils import get_comparison_data, load_npy_file, kl_config

# Assign unique values to each model
model_values = {
    "EleutherAI/pythia-70m-deduped": 0.1,
    "EleutherAI/pythia-410m-deduped": 0.5,
    "EleutherAI/pythia-1.4b-deduped": 0.9
}


# Function to map base model to lightness and target model to color
def get_color(base_model, target_model):
    base_value = model_values[base_model]
    target_value = model_values[target_model]
    lightness = 0.5 + 0.5 * base_value
    hue = 0.5 * target_value
    return plt.cm.hsv(hue)[:3] + (lightness,)


def plot_kl_scores(base_model_name, target_name):
    revisions = kl_config()["revisions"]
    kl_scores = []

    for revision in revisions:
        kl_data = get_comparison_data(load_npy_file(base_model_name, revision), target_name, revision)
        average_kl_score = np.mean(kl_data)
        kl_scores.append(average_kl_score)

    color = get_color(base_model_name, target_name)
    plt.plot(revisions, kl_scores, marker='o', linestyle='-', color=color,
             label=f'{X(base_model_name)} vs {X(target_name)}')


def X(string):
    return string.replace("EleutherAI/pythia-", "").replace("-deduped", "")


def plot_all_kl_scores():
    model_pairs = [
        ("EleutherAI/pythia-70m-deduped", "EleutherAI/pythia-410m-deduped"),
        ("EleutherAI/pythia-410m-deduped", "EleutherAI/pythia-70m-deduped"),
        ("EleutherAI/pythia-410m-deduped", "EleutherAI/pythia-1.4b-deduped"),
        ("EleutherAI/pythia-1.4b-deduped", "EleutherAI/pythia-410m-deduped"),
        ("EleutherAI/pythia-70m-deduped", "EleutherAI/pythia-1.4b-deduped"),
        ("EleutherAI/pythia-1.4b-deduped", "EleutherAI/pythia-70m-deduped")
    ]

    plt.figure(figsize=(12, 8))

    for base_model_name, target_name in model_pairs:
        plot_kl_scores(base_model_name, target_name)

    plt.xlabel('Revisions')
    plt.ylabel('Average KL Score')
    plt.title('Average KL Score vs Revisions for Different Model Comparisons')
    plt.legend()
    plt.grid(True)
    plt.savefig('../graphics/average_kl_vs_revisions.png')


plot_all_kl_scores()
