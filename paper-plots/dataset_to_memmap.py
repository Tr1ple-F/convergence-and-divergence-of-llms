import pandas as pd
import ipdb

df = pd.read_parquet('./n_test_2/train-00000-of-00009.parquet')
ipdb.set_trace()

import numpy as np

# Step 1: Compute total number of token_ids
total_tokens = sum(len(row) for row in df["token_ids"])

# Step 2: Create a memmap file for uint16
filename = "n_test_2/token_ids_uint16.dat"
memmap_array = np.memmap(
    filename,
    dtype=np.uint16,
    mode="w+",
    shape=(total_tokens,)
)

# Step 3: Write all token_ids into the memmap
offset = 0
for row in df["token_ids"]:
    length = len(row)
    memmap_array[offset:offset + length] = row
    offset += length

# Step 4: Flush to disk
memmap_array.flush()

print(f"Successfully saved {total_tokens} token_ids to '{filename}' as uint16 memmap.")
