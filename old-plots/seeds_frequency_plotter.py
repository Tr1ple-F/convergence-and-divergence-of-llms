import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import sys
import json
import numpy as np

# Load the dataframe
frequency_count = np.load("../frequency_count.npy")

print(np.sum(frequency_count, axis=1))

input_text = np.load(f'../working_dir/{sys.argv[1]}/input_text_encoded.npy')
