import matplotlib.pyplot as plt
import numpy as np

from utils import load_npy_file, kl_config, tokenized_text, get_comparison_data

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def moving_variance(data, moving_avg, window_size):
    squared_diff = (data[window_size - 1:] - moving_avg) ** 2
    return moving_average(squared_diff, window_size)

def plot_kl(tokenized_text, base_model_name, base_revision, target_name, target_revision, data, window_size=20):
    x_indices = np.arange(len(tokenized_text))
    y_scores = data.flatten()
    avg_kl_score = np.mean(y_scores)

    y_moving_avg = moving_average(y_scores, window_size)
    y_moving_var = moving_variance(y_scores, y_moving_avg, window_size)
    f = plt.figure(figsize=(40, 18))
    plt.plot(x_indices, y_scores, '-o', color='blue', label='KL Score')
    plt.plot(x_indices[window_size - 1:], y_moving_avg, color='red', linewidth=2, label='Moving Average')
    plt.plot(x_indices[window_size - 1:], y_moving_var, color='green', linewidth=2, label='Moving Variance')

    for idx, text in enumerate(tokenized_text):
        if text == '<|endoftext|>':
            plt.axvline(x=idx, color='green', linestyle='--')

    clean_base = base_model_name.replace("EleutherAI/pythia-", "").replace("-deduped", "")
    clean_target = target_name.replace("EleutherAI/pythia-", "").replace("-deduped", "")

    plt.xlabel("Index")
    plt.ylabel("KL Score")
    plt.title(
        f"{clean_base} # {base_revision.replace('step', '')} vs. {clean_target} # {target_revision.replace('step', '')}")

    plt.yticks(np.arange(0, 36, 2))
    plt.ylim(0, 34)

    textstr = f'Average KL Score: {avg_kl_score:.2f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=14,
             verticalalignment='top', bbox=props)

    plt.legend()
    plt.tight_layout()
    plt.savefig(
        f"../graphics/individual_kl/{clean_base}_{base_revision.replace('step', '')}_vs_{clean_target}_{target_revision.replace('step', '')}_kl.png")
    f.clear()
    plt.clf()
    plt.close(f)


def plot_surprisal(tokenized_text, base_model_name, base_revision, data, window_size=20):
    x_indices = np.arange(len(tokenized_text))
    y_scores = data.flatten()
    avg_surprise = np.mean(y_scores)
    y_moving_avg = moving_average(y_scores, window_size)

    f = plt.figure(figsize=(40, 18))
    plt.plot(x_indices, y_scores, '-o', color='blue')
    plt.plot(x_indices[window_size - 1:], y_moving_avg, color='red', linewidth=2, label='Moving Average')

    for idx, text in enumerate(tokenized_text):
        if text == '<|endoftext|>':
            plt.axvline(x=idx, color='green', linestyle='--')

    plt.yticks(np.arange(0, 15, 1))
    plt.ylim(0, 14)

    clean_base = base_model_name.replace("EleutherAI/pythia-", "").replace("-deduped", "")

    textstr = f'Average Surprisal Score: {avg_surprise:.2f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=14,
             verticalalignment='top', bbox=props)

    plt.xlabel("Index")
    plt.ylabel("Suprisal Score")
    plt.title(f"{clean_base} # {base_revision.replace('step', '')} Suprisal")
    plt.tight_layout()
    plt.savefig(f"../graphics/individual_surprisal/{clean_base}_{base_revision.replace('step', '')}_Suprisal.png")
    f.clear()
    plt.clf()
    plt.close(f)

for base_model_name in kl_config()['model_names']:
    for base_revision in kl_config()['revisions']:
        surprisal = load_npy_file(base_model_name, base_revision, "-", "surprisal")
        plot_surprisal(tokenized_text()[1:], base_model_name, base_revision, surprisal)
        for target_name in kl_config()['model_names']:
            for target_revision in kl_config()['revisions']:
                comparison_data = get_comparison_data(load_npy_file(base_model_name, base_revision), target_name,
                                                      target_revision)
                plot_kl(tokenized_text()[1:], base_model_name, base_revision, target_name, target_revision,
                        comparison_data)  # model 1 = P
