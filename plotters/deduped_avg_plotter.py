import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import deduped_config, strip

config = deduped_config()
revisions = config['revisions']
models = config['model_names']
number_of_steps = len(revisions)
number_of_models = len(models)
last_step = revisions[-1]

# Idea 1
# x = training step
# y = multiple KLs, different model size
for model in models:
    kl_data = np.load(f'../results/deduped/{model.replace("/", "-")}-{last_step}-kl.npy')
    selected_data = np.average(kl_data, axis=1)

    data = []
    for i, value in enumerate(selected_data):
        data.append({'Training Step': strip(revisions[i % number_of_steps]), 'Model': strip(models[int(i / number_of_steps)]), 'KL Value': value})
    df = pd.DataFrame(data)

    sns.lineplot(x='Training Step', y='KL Value', hue='Model', data=df, estimator=None)

    plt.savefig(f'../graphics/across_training_{model.replace("/", "-")}.png')
    plt.close()

# Idea 2
# x = model_size
# y = multiple KLs, all same model, different training steps
for model in models:
    kl_data = np.load(f'../results/deduped/{model.replace("/", "-")}-{last_step}-kl.npy')
    selected_data = np.average(kl_data, axis=1)

    data = []
    for i, value in enumerate(selected_data):
        data.append({'Training Step': strip(revisions[i % number_of_steps]), 'Model': strip(models[int(i / number_of_steps)]), 'KL Value': value})
    df = pd.DataFrame(data)

    sns.lineplot(x='Model', y='KL Value', hue='Training Step', data=df, estimator=None)

    plt.savefig(f'../graphics/across_model_{model.replace("/", "-")}.png')
    plt.close()
