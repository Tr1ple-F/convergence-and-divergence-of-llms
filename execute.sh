#!/bin/bash

# Check if a working directory argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <working_directory>"
    exit 1
fi

# Assign the working directory argument
WORKING_DIR="$1"

# Step 1: Run python tokenize_util inside ./run-stage
cd ./run-stage || { echo "Directory ./run-stage not found"; exit 1; }
python tokenize_util.py "$WORKING_DIR" || { echo "Failed to run tokenize_util"; exit 1; }

# Step 2: Run pos_tagger.py in the root directory
cd .. || exit
python pos_tagger.py "$WORKING_DIR" || { echo "Failed to run pos_tagger.py"; exit 1; }

# Step 3: Run python runner.py inside ./run-stage
cd ./run-stage || { echo "Directory ./run-stage not found"; exit 1; }
python runner.py "$WORKING_DIR" || { echo "Failed to run runner.py"; exit 1; }

# Step 4: Run python deduped_kl_torch.py and deduped_stats.py in ./stats-stage
cd ../stats-stage || { echo "Directory ./stats-stage not found"; exit 1; }
python deduped_kl_torch.py "$WORKING_DIR" || { echo "Failed to run deduped_kl_torch.py"; exit 1; }
python deduped_stats.py "$WORKING_DIR" || { echo "Failed to run deduped_stats.py"; exit 1; }

# Step 5: Tar zip the folder ./results/deduped
cd .. || exit
timestamp=$(date +"%Y-%m-%d-%H%M%S")
output_file="run-${timestamp}.tar.gz"
tar -czvf "$output_file" ./$WORKING_DIR/results/deduped || { echo "Failed to create tar.gz file"; exit 1; }

echo "Process completed successfully. Output file: $output_file"
