import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys

# Load the dataframe
probabilities = np.load(f'../working_dir/{sys.argv[1]}/probabilities/EleutherAI-pythia-{sys.argv[2]}/step{sys.argv[3]}/probabilities.npy')[:100, :50277]

# Generate frames
for i in range(probabilities.shape[0]):
    plt.figure(figsize=(10, 6))
    plt.plot(np.exp(probabilities[i]))
    plt.xlabel('Values')
    plt.ylabel('exp(Value)')
    plt.title(f'Frame {i+1}')
    plt.savefig(f'../working_dir/{sys.argv[1]}/output/frames/frame_{i+1}.png')
    plt.close()
