import matplotlib.pyplot as plt
import numpy as np
import json

with open("../kl_config.json") as f:
    kl_config = json.load(f)

def create_kl_grid():
    # Load all .npy files ending with '_kl'
    data = np.zeros((45, 45))
    i = 0
    ticks = []
    for base_model_name in kl_config['model_names']:
        for base_revision in kl_config['revisions']:
            # Load the file and calculate the averages
            loaded_data = np.load(f'../results/{base_model_name.replace("/", "-")}_{base_revision}_kl.npy')

            averages = np.mean(loaded_data, axis=1)

            # Insert the averages into the data array
            data[i, :i] = averages[:i]
            data[i, i+1:] = averages[i:]
            i += 1
            ticks.append(f"{base_model_name.replace('EleutherAI/pythia-', '').replace('-deduped', '')}-{base_revision}")

    # Set the diagonal to NaN
    np.fill_diagonal(data, 0)

    # Get the min and max values for the color scale
    vmin = np.nanmin(data)
    vmax = np.nanmax(data)

    # Adjust the figure size to make cells larger
    fig, ax = plt.subplots(figsize=(15, 15))
    cax = ax.matshow(np.log1p(data), cmap='gray', vmin=np.log1p(vmin), vmax=np.log1p(vmax))
    fig.colorbar(cax)
    plt.xticks(range(45), ticks, rotation=45)
    plt.yticks(range(45), ticks)
    plt.savefig('../graphics/kl_grid.png')
    plt.close('all')

def create_moby_dick_grid():
    # Load all .npy files ending with '_kl'
    data = np.zeros((45, 45))
    i = 0
    ticks = []
    for base_model_name in kl_config['model_names']:
        for base_revision in kl_config['revisions']:
            # Load the file and calculate the averages
            loaded_data = np.load(f'../results/{base_model_name.replace("/", "-")}_{base_revision}_kl.npy')

            averages = np.mean(loaded_data[:, 2272:2636], axis=1) - np.mean(loaded_data, axis=1) + 2

            # Insert the averages into the data array
            data[i, :i] = averages[:i]
            data[i, i+1:] = averages[i:]
            i += 1
            ticks.append(f"{base_model_name.replace('EleutherAI/pythia-', '').replace('-deduped', '')}-{base_revision}")

    # Set the diagonal to NaN
    np.fill_diagonal(data, 0)

    # Get the min and max values for the color scale
    vmin = np.nanmin(data)
    vmax = np.nanmax(data)

    # Adjust the figure size to make cells larger
    fig, ax = plt.subplots(figsize=(15, 15))
    cax = ax.matshow(np.log1p(data), cmap='gray', vmin=np.log1p(vmin), vmax=np.log1p(vmax))
    fig.colorbar(cax)

    plt.xticks(range(45), ticks, rotation=45)
    plt.yticks(range(45), ticks)
    plt.savefig('../graphics/moby_grid.png')
    plt.close('all')

def create_code_grid():
    # Load all .npy files ending with '_kl'
    data = np.zeros((45, 45))
    i = 0
    ticks = []
    for base_model_name in kl_config['model_names']:
        for base_revision in kl_config['revisions']:
            # Load the file and calculate the averages
            loaded_data = np.load(f'../results/{base_model_name.replace("/", "-")}_{base_revision}_kl.npy')

            averages = np.mean(loaded_data[:, 3500:], axis=1) - np.mean(loaded_data, axis=1) + 2

            # Insert the averages into the data array
            data[i, :i] = averages[:i]
            data[i, i+1:] = averages[i:]
            i += 1
            ticks.append(f"{base_model_name.replace('EleutherAI/pythia-', '').replace('-deduped', '')}-{base_revision}")

    # Set the diagonal to NaN
    np.fill_diagonal(data, 0)

    # Get the min and max values for the color scale
    vmin = np.nanmin(data)
    vmax = np.nanmax(data)

    # Adjust the figure size to make cells larger
    fig, ax = plt.subplots(figsize=(15, 15))
    cax = ax.matshow(np.log1p(data), cmap='gray', vmin=np.log1p(vmin), vmax=np.log1p(vmax))
    fig.colorbar(cax)

    plt.xticks(range(45), ticks, rotation=45)
    plt.yticks(range(45), ticks)
    plt.savefig('../graphics/code_grid.png')
    plt.close('all')

create_kl_grid()
create_moby_dick_grid()
create_code_grid()
