import json
import sys

def deduped_config():
    with open(f'../working_dir/{sys.argv[1]}/deduped_config.json') as f:
        return json.load(f)

def seeds_config():
    with open(f'../working_dir/{sys.argv[1]}/seeds_config.json') as f:
        return json.load(f)

def tagged_tokens():
    with open(f'../working_dir/{sys.argv[1]}/pos_tagged_tokens.json', 'r') as f:
        return json.load(f)

def pos_tags():
    with open('../common/pos_tags.json', 'r') as f:
        return json.load(f)

def strip(name):
    if "google" in name:
        return name.replace("google/multiberts", "multiberts")
    if "EleutherAI" in name:
        return name.replace("EleutherAI/pythia-", "").replace("-deduped", "")
    if "step_" in name:
        return name.replace("step_", "").replace("k", "000")
    if "step" in name:
        return name.replace("step", "")
    return name
