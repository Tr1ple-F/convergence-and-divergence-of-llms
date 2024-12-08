import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import sys
import json
import numpy as np

with open(f'../working_dir/{sys.argv[1]}/pos_tagged_tokens.json', 'r') as f:
    pos_tags = json.load(f)

pos_tags_1 = [x[1] for x in pos_tags]

pos_tags_1_counts = pd.Series(pos_tags_1).value_counts()
pos_tags_1_counts.drop('UNK', inplace=True)
print(pos_tags_1_counts)

plt.figure(figsize=(10, 6))
sns.barplot(x=pos_tags_1_counts.index, y=pos_tags_1_counts.values)
plt.title('Distribution of PoS Tags')
plt.xlabel('PoS Tag')
plt.ylabel('Frequency')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(f'../working_dir/{sys.argv[1]}/output/pos_tags_distribution.png')
plt.close()
