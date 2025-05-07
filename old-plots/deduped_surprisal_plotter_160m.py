import sys
import numpy as np
import matplotlib.pyplot as plt
from utils import deduped_config, strip

models = deduped_config()['model_names']
revisions = deduped_config()['revisions']

def plot(model_index, revision_index):
    model = models[model_index]
    revision = revisions[revision_index]

    surprisal_data = np.load(f'../working_dir/{sys.argv[1]}/results/deduped/{model.replace("/", "-")}-{revision}-surprisal.npy')

    plt.figure(figsize=(10, 6))
    plt.plot(surprisal_data)
    plt.savefig(f'../working_dir/{sys.argv[1]}/output/surprisal_plot_{strip(model)}_{strip(revision)}.png')
    plt.close()

plot(0,0)
plot(0,-1)
plot(1,0)
plot(1,-1)
plot(2,0)
plot(2,-1)
