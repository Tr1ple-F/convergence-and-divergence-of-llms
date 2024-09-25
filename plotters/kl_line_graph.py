import numpy as np
import matplotlib.pyplot as plt
from utils import kl_config

# Function to plot average KL values
def plot_average_kl_values(kl_data, model_name, revision):
    kl_averages = np.mean(kl_data, axis=1)
    plt.figure(figsize=(10, 6))
    plt.plot(kl_averages, marker='o')
    plt.title(f'Average KL Values for {model_name} - {revision}')
    plt.xlabel('Comparison Index')
    plt.ylabel('Average KL Value')
    plt.grid(True)
    plt.savefig(f'../graphics/{model_name.replace("EleutherAI/pythia-", "").replace("-deduped", "")}-{revision}-line.png')
    plt.close('all')

# Load and plot data
for base_model_name in kl_config()['model_names']:
    for base_revision in kl_config()['revisions']:
        # Load the file and calculate the averages
        file_path = f'../results/{base_model_name.replace("/", "-")}_{base_revision}_kl.npy'
        loaded_data = np.load(file_path)
        plot_average_kl_values(loaded_data, base_model_name, base_revision)
