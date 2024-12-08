import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import deduped_config, pos_tags, tagged_tokens, strip

index_of_model = int(sys.argv[2])
index_of_revision = int(sys.argv[3])

config = deduped_config()
model_names = config['model_names']
revisions = config['revisions']
length = len(revisions)

print(f"Model: {model_names[index_of_model]} - Revision: {revisions[index_of_revision]}")

kl_base = np.load(f'../working_dir/{sys.argv[1]}/results/deduped/{model_names[index_of_model].replace("/", "-")}-{revisions[index_of_revision]}-kl.npy')
print("KL Base Shape:", kl_base.shape)
# Compare e.g. 12b 128k to 12b 143k, 6.9b 128k, 6.9b 143k

model_indices = [index_of_model*length + index_of_revision + 2, (index_of_model-2)*length + index_of_revision, (index_of_model-2)*length + index_of_revision + 2]
print("Model Indices:", model_indices)
kl_selection = kl_base[model_indices, :]
print("KL Selection Shape:", kl_selection.shape)

pos_kl = []
non_null_pos_tags = []
reference_kl = np.mean(kl_selection[:, :], axis=1)
for i, pos_tag in enumerate(pos_tags()):
    pos_indices = [idx for idx, tag in enumerate(tagged_tokens()) if tag[1] == pos_tag["tag"]]
    if len(pos_indices) == 0:
        continue
    else:
        non_null_pos_tags.append(pos_tag)
    kl_averages = np.mean(kl_selection[:, pos_indices], axis=1)
    pos_kl.append(kl_averages)

data = []
row = {'Model': strip(model_names[index_of_model]) + " - " + strip(revisions[index_of_revision + 2]), 'KL Average - Ref': reference_kl[0]}
for j, pos_tag in enumerate(non_null_pos_tags):
    row[f'KL Average - {pos_tag["tag"]}'] = pos_kl[j][0]
data.append(row)
row = {'Model': strip(model_names[index_of_model-2]) + " - " + strip(revisions[index_of_revision]), 'KL Average - Ref': reference_kl[1]}
for j, pos_tag in enumerate(non_null_pos_tags):
    row[f'KL Average - {pos_tag["tag"]}'] = pos_kl[j][1]
data.append(row)
row = {'Model': strip(model_names[index_of_model-2]) + " - " + strip(revisions[index_of_revision + 2]), 'KL Average - Ref': reference_kl[2]}
for j, pos_tag in enumerate(non_null_pos_tags):
    row[f'KL Average - {pos_tag["tag"]}'] = pos_kl[j][2]
data.append(row)

df = pd.DataFrame(data)
print(df)

# Create a bar plot
df_melted = df.melt(id_vars=['Model'], var_name='Metric', value_vars=[f'KL Average - {x}' for x in ["Ref", "VB", "JJ", "IN", "NN", "NNS", "NNP", "NNPS", "VBG", "VBN", "VBD", "VBZ"]], value_name='KL Average')
sns.barplot(data=df_melted, x='Metric', y='KL Average', hue='Model')
plt.title('KL Average by Model and Metric')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'../working_dir/{sys.argv[1]}/output/analyzer_kl_average_barplot.png')
plt.close()
