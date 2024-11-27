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
# python tokenize_util.py "$WORKING_DIR" || { echo "Failed to run tokenize_util"; exit 1; }

# Step 2: Run pos_tagger.py inside ./run-stage
# python pos_tagger.py "$WORKING_DIR" || { echo "Failed to run pos_tagger.py"; exit 1; }

# Step 3: Run python runner_seeds.py inside ./run-stage
python runner_seeds.py "$WORKING_DIR" || { echo "Failed to run runner_seeds.py"; exit 1; }

# Step 4: Run python seeds_kl_torch.py and seeds_stats.py in ./stats-stage
cd ../stats-stage || { echo "Directory ./stats-stage not found"; exit 1; }
for index in {0..4}; do
    tmux new-session -d -s "kl_torch_$index" \
        "CUDA_VISIBLE_DEVICES=$index python seeds_kl_torch.py \"\$WORKING_DIR\" $index || { echo \"Failed to run seeds_kl_torch.py for index $index\"; exit 1; }"
    echo "Started Tmux session 'kl_torch_$index'"
done

# Create and start Tmux session for seeds_stats.py
tmux new-session -d -s "stats" \
    "python seeds_stats.py \"\$WORKING_DIR\" || { echo \"Failed to run seeds_stats.py\"; exit 1; }"
echo "Started Tmux session 'stats'"

# Step 5: Tar zip the folder ./results/seeds
cd .. || exit
timestamp=$(date +"%Y-%m-%d-%H%M%S")
output_file="run-${timestamp}.tar.gz"
tar -czvf "$output_file" ./working_dir/$WORKING_DIR/results/seeds || { echo "Failed to create tar.gz file"; exit 1; }

echo "Process completed successfully. Output file: $output_file"
