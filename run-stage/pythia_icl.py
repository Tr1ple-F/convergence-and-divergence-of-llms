import os
import sys
import json
from time import sleep

import torch
import pandas as pd
import torch.nn.functional as F
from transformers import GPTNeoXForCausalLM, AutoTokenizer

def calculate_surprisal(probabilities, correct_indices):
    return -probabilities[torch.arange(probabilities.shape[0]), correct_indices]

def run_icl_for_model(model_n, revision, seed, input_text):
    model_name = f"{model_n}-seed{seed}"
    print(f"Running ICL for {model_name}, revision {revision}")

    cache_dir = f"./{model_name.replace('/', '-')}/{revision}"

    try:
        model = GPTNeoXForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            cache_dir=cache_dir,
        )
    except:
        print(f"Could not load model {model_name}")
        return []

    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-14m-seed1",
        revision="step128",
        cache_dir=f"./EleutherAI-pythia-14m-seed1/step128",
        bos_token='<|endoftext|>',
        eos_token='<|endoftext|>',
        add_bos_token=True,
        add_eos_token=True
    )

    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    inputs = []
    for snippet in input_text.split("<|endoftext|>")[:-1]: # Skip last
        tokenized = tokenizer(snippet, return_tensors="pt")
        tokenized['input_ids'] = tokenized['input_ids'][:, :550]
        inputs.append(tokenized)

    inputs_loaded = []
    if torch.cuda.is_available():
        for snippet in inputs:
            inputs_loaded.append({k: v.cuda() for k, v in snippet.items()})

    scores = []

    with torch.no_grad():
        for sample in inputs_loaded:
            input_ids_ = sample['input_ids']
            logits = model(input_ids_).logits
            probabilities = F.log_softmax(logits, dim=-1).reshape(-1, logits.shape[-1])
            correct_indices = input_ids_[:, 1:]
            surprisal = calculate_surprisal(probabilities[:-1], correct_indices)[0]
            icl_score = surprisal[500] - surprisal[50]
            row = {
                'Model': model_n,
                'Training Step': revision,
                'Seed': seed,
                'ICL': icl_score.item()
            }
            scores.append(row)

    return scores

def main():
    experiment_id = sys.argv[1]

    with open(f'../working_dir/{experiment_id}/seeds_config.json', 'r') as f:
        config = json.load(f)

    model_names = config['model_names']
    revisions = config['revisions']
    seeds = config['seeds']

    with open(f'../working_dir/{experiment_id}/input_text.txt', 'r', encoding="utf8") as file:
        input_text = file.read()

    data = []

    for model_name in model_names:
        for revision in revisions:
            for i in seeds:
                icl_scores = run_icl_for_model(model_name, revision, i, input_text)
                while not icl_scores:
                    sleep(5)
                    icl_scores = run_icl_for_model(model_name, revision, i, input_text)
                data = data + icl_scores

    df = pd.DataFrame(data)
    output_dir = f'../working_dir/{experiment_id}/output'
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, 'seeds_icl.csv'), index=False)
    print("ICL scores saved.")

if __name__ == "__main__":
    main()
