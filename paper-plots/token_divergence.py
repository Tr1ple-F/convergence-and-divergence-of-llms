import pandas as pd
import sys
import json

df = pd.read_csv(f'../working_dir/{sys.argv[1]}/output/seeds_token_divergence_dataframe.csv')

filtered_df = df[
    (df['Model'] == df['Model 2']) &
    (df['Seed 1'] != df['Seed 2']) &
    (df['Training Step'] == df['Training Step 2']) &
    (df['POS'] != 'UNK')
]

# Step 2: Group by Token and compute average KL
avg_kl_by_token = filtered_df.groupby('Token')['KL'].mean().reset_index()

# Step 3: Sort by KL to find highest, lowest, and most middle
sorted_kl = avg_kl_by_token.sort_values('KL').reset_index(drop=True)

# Get the lowest, highest, and most middle tokens
lowest = sorted_kl.iloc[0:10]
df_len = len(sorted_kl)
highest = sorted_kl.iloc[df_len - 10 : df_len]
mid = df_len // 2
middle = sorted_kl.iloc[mid - 5 : mid + 5]

with open("../common/vocab.json", "r") as f:
    vocab = json.load(f)

def pretty_print(subset, label):
    print(f"\n{label} KL Tokens:")
    for index, entry in subset.iterrows():
        token_id = int(entry['Token'])
        token_str = vocab[token_id]
        kl = entry['KL']
        print(f"  Token ID     : {token_id}")
        print(f"  Text         : {repr(token_str)}")
        print(f"  Avg KL Diverg: {kl:.6f}")
        print()

pretty_print(lowest, "Lowest")
pretty_print(middle, "Middle")
pretty_print(highest, "Highest")