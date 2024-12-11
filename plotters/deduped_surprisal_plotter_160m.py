import sys
import numpy as np
import matplotlib.pyplot as plt
from utils import deduped_config, strip

models = deduped_config()['model_names']
revisions = deduped_config()['revisions']

model1 = models[1]
revision1 = revisions[-1]

surprisal_data = np.load(f'../working_dir/{sys.argv[1]}/results/deduped/{model1.replace("/", "-")}-{revision1}-surprisal.npy')

plt.figure(figsize=(10, 6))
plt.plot(surprisal_data)
plt.savefig(f'../working_dir/{sys.argv[1]}/output/surprisal_plot_160m_143k.png')
plt.close()
