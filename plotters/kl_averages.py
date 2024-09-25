import matplotlib.pyplot as plt
import numpy as np

from utils import get_comparison_data, load_npy_file, kl_config


def plot_kl_scores(base_model_name, target_name):
    revisions = kl_config()["revisions"]
    kl_scores = []

    for revision in revisions:
        kl_data = get_comparison_data(load_npy_file(base_model_name, revision), target_name, revision)
        average_kl_score = np.mean(kl_data)
        kl_scores.append(average_kl_score)

    plt.plot(revisions, kl_scores, marker='o', linestyle='-', label=f'{X(base_model_name)} vs {X(target_name)}')


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
    plt.savefig('./graphics/average_kl_vs_revisions.png')


plot_all_kl_scores()
