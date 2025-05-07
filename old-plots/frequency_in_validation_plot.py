import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import sys
import json

frequency_count = np.load("../frequency_count.npy")
# open vocab.json
with open("../common/vocab.json", "r") as f:
    vocab = json.load(f)

# Create a DataFrame for better handling
df = pd.DataFrame(frequency_count, columns=["frequency"])

# Sort the DataFrame by frequency in descending order
df = df.sort_values(by="frequency", ascending=False)
df["id"] = range(1, len(df) + 1)
df["token"] = [("'" + vocab[i] + "'") for i in df.index]
df = df[df["frequency"] >= 1]
print(df)
print(df.head(20))

# Plot the top x tokens by frequency
plt.figure(figsize=(12, 6))
sns.lineplot(x="id", y="frequency", data=df)
plt.xlabel("Token ranked by frequency")
plt.ylabel("Frequency")
# plt.yscale("log")
plt.xscale("log")
plt.title(f"Frequency of Tokens in Validation Set by most frequent")
plt.xticks(rotation=90)
plt.savefig(f'../working_dir/{sys.argv[1]}/output/frequency_in_validation.png')
plt.close()
