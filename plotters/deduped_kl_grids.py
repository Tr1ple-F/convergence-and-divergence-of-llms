import matplotlib.pyplot as plt
import numpy as np
import sys
from utils import deduped_config

def create_kl_grid_deduped():
    data = np.zeros((72, 72))
    i = 0
    ticks = []
    for base_model_name in deduped_config()['model_names']:
        for base_revision in deduped_config()['revisions']:
            loaded_data = np.load(f'../working_dir/{sys.argv[1]}/results/deduped/{base_model_name.replace("/", "-")}-{base_revision}-kl.npy')
            averages = np.mean(loaded_data, axis=1)
            data[i, :] = averages
            i += 1
            ticks.append(f"{base_model_name.replace('EleutherAI/pythia-', '').replace('-deduped', '')}-{base_revision}")

    # Get the min and max values for the color scale
    vmin = np.nanmin(data)
    vmax = np.nanmax(data)

    # Adjust the figure size to make cells larger
    fig, ax = plt.subplots(figsize=(15, 15))
    cax = ax.matshow(np.log1p(data), cmap='gray', vmin=np.log1p(vmin), vmax=np.log1p(vmax))
    fig.colorbar(cax)
    plt.xticks(range(72), ticks, rotation=45)
    plt.yticks(range(72), ticks)
    plt.savefig(f'../working_dir/{sys.argv[1]}/output/kl_grid_deduped.png')
    plt.close('all')

create_kl_grid_deduped()
