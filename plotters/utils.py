import numpy as np
import json
import os

def deduped_config():
    with open("./deduped_config.json") as f:
        return json.load(f)

def seeds_config():
    with open("./seeds_config.json") as f:
        return json.load(f)

def tokenized_text():
    with open("../input_text_tokenized_old.json") as f:
        return json.load(f)

# Function to load the appropriate .npy file
def load_npy_file(model_name, revision, middle="_", appendix="kl"):
    model_name_sanitized = model_name.replace('/', '-')
    filename = f"{model_name_sanitized}{middle}{revision}{middle}{appendix}.npy"
    filepath = os.path.join("../results/deduped/", filename)

    if os.path.exists(filepath):
        data = np.load(filepath)
        return data
    else:
        raise FileNotFoundError(f"The file {filename} does not exist in the 'results' folder.")

def find_comparison_index(model_name, revision):
    try:
        model_names = deduped_config()['model_names']
        revisions = deduped_config()['revisions']
        model_index = model_names.index(model_name)
        revision_index = revisions.index(revision)
        return model_index * len(revisions) + revision_index
    except ValueError:
        return -1

def get_comparison_data(data, target_model_name, target_revision):
    comparison_index = find_comparison_index(target_model_name, target_revision)
    return data[comparison_index, :]
