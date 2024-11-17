import json
import sys

import numpy as np
import pandas as pd

# Load configuration from JSON for model sizes and steps
with open(f'../working_dir/{sys.argv[1]}/deduped_config.json', 'r') as f:
    config = json.load(f)

# Load the JSON file containing tokenized data
with open(f'../working_dir/{sys.argv[1]}/input_text_tokenized.json', 'r') as f:
    tokenized_data = json.load(f)

# Change #1: Create a new DataFrame for word IDs and save it
word_id_df = pd.DataFrame(columns=['id', 'w<t', 'w=t'])
for j in range(len(tokenized_data)):
    previous_context = tokenized_data[:j] if j > 0 else []
    word_id_df = pd.concat([word_id_df, pd.DataFrame([{
        'id': j,
        'w<t': previous_context,
        'w=t': tokenized_data[j]
    }])], ignore_index=True)

# Save word IDs to a CSV file
word_id_df.to_csv(f'../working_dir/{sys.argv[1]}/word_ids.csv', index=False)

surprisals = []
data = []

for size in config["sizes"]:
    for step in config["steps"]:
        surprisals.append(np.load(f'../working_dir/{sys.argv[1]}/results/deduped/EleutherAI-pythia-{size}-deduped-step{step}-surprisal.npy'))

model_index = 0
for size in config["sizes"]:
    for step in config["steps"]:
        # Construct file names based on size and step
        kl_filename = f'../working_dir/{sys.argv[1]}/results/deduped/EleutherAI-pythia-{size}-deduped-step{step}-kl.npy'

        kl_values = np.load(kl_filename)

        kl_index = 0

        for size2 in config["sizes"]:
            for step2 in config["steps"]:
                surprisal_model1 = surprisals[model_index]
                surprisal_model2 = surprisals[kl_index]

                for j in range(kl_values.shape[1]):
                    data.append({
                        'model1': size + "-step" + step,
                        'model2': size2 + "-step" + step2,
                        'word_id': j,
                        'kl': kl_values[kl_index, j],
                        'surprisal1': surprisal_model1[j],
                        'surprisal2': surprisal_model2[j]
                    })

                kl_index += 1

        model_index += 1

# Save the combined DataFrame to a CSV file
df = pd.DataFrame(data)
df.to_csv(f'../working_dir/{sys.argv[1]}/dataframe.csv', index=False)
