import matplotlib.pyplot as plt
import numpy as np
import json

from utils import deduped_config, seeds_config

# Load POS tags from JSON file
with open('../common/pos_tagged_tokens.json', 'r') as f:
    pos_tagged_tokens = json.load(f)

with open('./pos_tags.json', 'r') as f:
    unique_pos_tags = json.load(f)

def create_kl_grid_deduped_by_pos():
    for pos_id,pos_tag in enumerate(unique_pos_tags):
        # print(pos_tag["tag"]) -- This works
        data = np.zeros((72, 72))
        i = 0
        ticks = []
        for base_model_name in deduped_config()['model_names']:
            for base_revision in deduped_config()['revisions']:
                loaded_data = np.load(f'../results/deduped/{base_model_name.replace("/", "-")}-{base_revision}-kl.npy')
                pos_indices = [idx for idx, tag in enumerate(pos_tagged_tokens) if tag[1] == pos_tag["tag"]]
                averages = np.mean(loaded_data[:, pos_indices], axis=1)
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
        plt.savefig(f'../graphics/kl_grid_deduped_{pos_id}.png')
        plt.close('all')

def create_kl_grid_seeds_by_pos():
    for pos_id,pos_tag in enumerate(unique_pos_tags):
        data = np.zeros((75, 75))
        i = 0
        ticks = []
        for base_model_name in seeds_config()['model_names']:
            for base_revision in seeds_config()['revisions']:
                for seed in [1, 5, 9]:
                    loaded_data = np.load(f'../results/seeds/{base_model_name.replace("/", "-")}-{base_revision}-seed{seed}-kl.npy')
                    pos_indices = [idx for idx, tag in enumerate(pos_tagged_tokens) if tag[1] == "PRP"]
                    averages = np.mean(loaded_data[:, pos_indices], axis=1)
                    data[i, :] = averages
                    i += 1
                    ticks.append(f"{base_model_name.replace('EleutherAI/pythia-', '').replace('-seed', '')}-{base_revision}")

        # Get the min and max values for the color scale
        vmin = np.nanmin(data)
        vmax = np.nanmax(data)

        # Adjust the figure size to make cells larger
        fig, ax = plt.subplots(figsize=(15, 15))
        cax = ax.matshow(np.log1p(data), cmap='gray', vmin=np.log1p(vmin), vmax=np.log1p(vmax))
        fig.colorbar(cax)
        plt.xticks(range(75), ticks, rotation=45)
        plt.yticks(range(75), ticks)
        plt.savefig(f'../graphics/kl_grid_seeds_{pos_id}.png')
        plt.close('all')

create_kl_grid_deduped_by_pos()
# create_kl_grid_seeds_by_pos()
for pos_id,pos_tag in enumerate(unique_pos_tags):
    print(f"{pos_id}: {pos_tag['tag']}")
