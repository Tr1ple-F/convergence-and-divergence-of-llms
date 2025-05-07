## Stages

### Setup

Initially we sampled from the pile validation set using the script `pile_sample.py`.
All scripts will work in the `working_dir` under the respective sampling environment (e.g. `sample1`).

### run-stage

This directory contains all scripts needed to run inference on training samples.  

- `tokenize_util.py` will tokenize the text samples
- `pos_tagger.py` will determine the part of speech tags for the input text
- `runner.py` will run inference on the text with the Pythia deduped models
- `runner_seeds.py` will run inference on the text with the Pythia standard models with seeds

### stats-stage

Using the outputs from the previous stage one can then calculate the KL between logits.

- `deduped_kl_torch.py` will calculate the KL between model runs
- `deduped_kl_uni.py` will calculate the KL against uniform and unigram distribution
- `deduped_stats.py` will calculate the surprisal values

The respective scripts exist for the seed runs.

### dataframes

The dataframes folder contains scripts to generate dataframes from the KL and surprisal outputs from the previous stage.
The dataframes will be stored under the working sample in the folder `output`.

### paper-plots

This folder contains the scripts to reproduce the plots in our paper.