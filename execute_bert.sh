#!/bin/bash

# Check if a working directory argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <working_directory>"
    exit 1
fi

# Assign the working directory argument
WORKING_DIR="$1"

# Step 3: Run python runner_seeds.py inside ./run-stage
cd ./run-stage || { echo "Directory ./run-stage not found"; exit 1; }
srun --gpus=1 --partition standard --unbuffered ../../venv/bin/python runner_bert.py "$WORKING_DIR" 0,1,2,3,4 || { echo "Failed to run runner_seeds.py"; exit 1; }

# Step 4: Run python seeds_kl_torch.py and seeds_stats.py in ./stats-stage
cd ../stats-stage || { echo "Directory ./stats-stage not found"; exit 1; }
srun --gpus=1 --partition standard --unbuffered ../../venv/bin/python bert_stats.py "$WORKING_DIR" || { echo "Failed to run seeds_stats.py"; exit 1; }
srun --gpus=1 --partition standard --unbuffered ../../venv/bin/python bert_uni.py "$WORKING_DIR" || { echo "Failed to run seeds_stats.py"; exit 1; }