import os
import sys
import json
from time import sleep
import glob
import itertools

import torch
import pandas as pd
import torch.nn.functional as F
from transformers import GPTNeoXForCausalLM, AutoTokenizer

import torch
import torch.nn.functional as F
import glob, os, json
from transformers import GPTNeoXForCausalLM, AutoTokenizer

def sentence_log_prob(model, tokenizer, sentence, device):
    """Compute log probability of a single sentence under the model on the given device."""
    enc = tokenizer(sentence, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)  # keep on device

    with torch.no_grad():
        logits = model(input_ids).logits  # [1, seq_len, vocab]
        log_probs = F.log_softmax(logits, dim=-1)

    # Shift: predict token t given <0..t-1>
    token_log_probs = log_probs[:, :-1, :].gather(
        2, input_ids[:, 1:].unsqueeze(-1)
    ).squeeze(-1)

    return token_log_probs.sum()  # scalar tensor on device


def run_blimp_for_model(model_n, revision, seed, blimp_dir):
    model_name = f"{model_n}-seed{seed}"
    print(f"Running BLiMP eval for {model_name}, revision {revision}")

    try:
        model = GPTNeoXForCausalLM.from_pretrained(
            f"./EleutherAI-pythia-raw/models/{model_name.replace('/', '-')}/{revision}/"
        )
    except Exception as e:
        print(f"Could not load model {model_name}: {e}")
        return []

    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-14m-seed1",
        revision="step128",
        cache_dir=f"./EleutherAI-pythia-14m-seed1/step128",
        bos_token='<|endoftext|>',
        eos_token='<|endoftext|>',
        add_bos_token=True,
        add_eos_token=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    results = []

    task_files = glob.glob(os.path.join(blimp_dir, "*.jsonl"))
    for task_file in task_files:
        task_name = os.path.splitext(os.path.basename(task_file))[0]

        with open(task_file, "r") as f:
            blimp_data = [json.loads(line) for line in f]

        # Preallocate tensor for probabilities
        probs_tensor = torch.zeros(len(blimp_data), 2, device=device)

        for idx, sample in enumerate(blimp_data):
            good_sent = sample["sentence_good"]
            bad_sent = sample["sentence_bad"]

            good_score = sentence_log_prob(model, tokenizer, good_sent, device)
            bad_score = sentence_log_prob(model, tokenizer, bad_sent, device)

            # Convert log-likelihoods to probability distribution
            scores = torch.stack([good_score, bad_score])  # tensor on device
            probs_tensor[idx] = F.softmax(scores, dim=0)

        # Average over all samples in the task
        avg_dist = probs_tensor.mean(dim=0)

        results.append(
            {
                "Model": model_n,
                "Training Step": revision,
                "Seed": seed,
                "Task": task_name,
                "GoodProb": avg_dist[0].item(),
                "BadProb": avg_dist[1].item(),
            }
        )

    return results

def compute_seed_kl(df):
    """Compute pairwise KL between seeds per model/revision/task."""
    rows = []
    for (model, step, task), group in df.groupby(["Model", "Training Step", "Task"]):
        seeds = group["Seed"].unique()
        for s1, s2 in itertools.combinations(seeds, 2):
            p = group[group["Seed"] == s1][["GoodProb", "BadProb"]].values[0]
            q = group[group["Seed"] == s2][["GoodProb", "BadProb"]].values[0]
            p = torch.tensor(p, dtype=torch.float32)
            q = torch.tensor(q, dtype=torch.float32)
            kl = F.kl_div(q.log(), p, reduction="sum").item()  # KL(p||q)
            rows.append(
                {
                    "Model": model,
                    "Training Step": step,
                    "Task": task,
                    "Seed1": s1,
                    "Seed2": s2,
                    "KL": kl,
                }
            )
    return pd.DataFrame(rows)


def main():
    experiment_id = sys.argv[1]

    with open(f"../working_dir/{experiment_id}/seeds_config.json", "r") as f:
        config = json.load(f)

    model_names = config["model_names"]
    revisions = config["revisions"]
    seeds = config["seeds"]

    blimp_dir = f"../../blimp/data"

    data = []

    for model_name in model_names:
        for revision in revisions:
            for i in seeds:
                results = run_blimp_for_model(model_name, revision, i, blimp_dir)
                while not results:
                    sleep(5)
                    results = run_blimp_for_model(model_name, revision, i, blimp_dir)
                data.extend(results)

    output_dir = f"../working_dir/{experiment_id}/output"
    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame(data)
    df.to_csv(os.path.join(output_dir, "blimp_distributions.csv"), index=False)
    kl_df = compute_seed_kl(df)
    kl_df.to_csv(os.path.join(output_dir, "blimp_seed_kl.csv"), index=False)

    print("BLiMP distributions and seed KL scores saved.")


if __name__ == "__main__":
    main()
