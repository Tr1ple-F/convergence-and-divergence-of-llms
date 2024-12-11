#!/bin/bash

# Check if a working directory argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <working_directory>"
    exit 1
fi

# Assign the working directory argument
WORKING_DIR="$1"

cd ./plotters || { echo "Directory ./plotters not found"; exit 1; }
echo "Running seeds to uni dataframe and plot"
python seeds_uni_dataframe.py "$WORKING_DIR" || { echo "Failed to run seeds_uni_dataframe"; exit 1; }
python seeds_uni_plot.py "$WORKING_DIR" || { echo "Failed to run seeds_uni_plot"; exit 1; }
