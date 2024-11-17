import json

def deduped_config():
    with open("./deduped_config.json") as f:
        return json.load(f)

def seeds_config():
    with open("./seeds_config.json") as f:
        return json.load(f)

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

def strip(name):
    if "EleutherAI" in name:
        return name.replace("EleutherAI/pythia-", "").replace("-deduped", "")
    if "step" in name:
        return name.replace("step", "")
    return name
