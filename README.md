## Reproducing the experiments

This repository contains a full pipeline to reproduce our experiments on training-time convergence of language models across random seeds and training steps. The workflow is:

1) Sample text from The Pile validation split
2) Prepare a `working_dir/<experiment_id>` and preprocess inputs (tokenize, POS-tags)
3) Run inference to produce token-level log-probabilities (`probabilities.npy`)
4) Reduce probabilities to analysis-ready arrays (pairwise KL, unigram/uniform KL, surprisals)
5) Build CSV dataframes
6) Plot the figures used in the paper

All intermediate and final artifacts are written under `working_dir/<experiment_id>`.

### 1) Sampling The Pile

We sample from The Pile validation set using the scripts at the project root. The code uses the HF dataset `pietrolesci/pile-validation`.

- `pile_sampler.py`: sample a small set of documents and write concatenated text to `working_dir/sample_text.txt`.
- `pile_sampler_icl.py`: sample longer snippets for ICL, writes `working_dir/icl_score/input_text.txt`.
- `pile_sampler_bert.py`: sample for BERT/MultiBERT experiments, writes `working_dir/input_text.txt`.
- `pile_sampler_frequency.py`: compute token frequency counts for Pythia tokenization → `frequency_count.npy`.
- `pile_sampler_frequency_bert.py`: compute token frequency counts for BERT tokenization → `frequency_count_bert.npy`.

From these counts you can build reference distributions used later:

- `unigram.py`: builds `unigram_dist.npy` and `uniform_dist.npy` from `frequency_count.npy`.
- `unigram_bert.py`: builds `unigram_dist_bert.npy` and `uniform_dist_bert.npy` from `frequency_count_bert.npy`.

References:
- The Pile: An 800GB Dataset of Diverse Text for Language Modeling ([paper](https://arxiv.org/abs/2101.00027))
- EleutherAI Pythia models ([HF hub](https://huggingface.co/EleutherAI))
- Google MultiBERTs ([HF hub](https://huggingface.co/google/multiberts-seed_0))
- BLiMP benchmark ([repo](https://github.com/alexwarstadt/blimp))

### 2) Working directory layout

For each experiment create a folder `working_dir/<experiment_id>` and place/configure:

- `input_text.txt`: the raw text to evaluate (e.g., from the samplers above). Sentences are usually separated by `<|endoftext|>` for Pythia and `[SEP][CLS]` for BERT-specific runs.
- `seeds_config.json`: models, revisions (steps), and seeds for seed-based runs (Pythia and BERT).
- `deduped_config.json`: models and revisions for deduped Pythia runs (not used in the paper, but code is provided).

During preprocessing and inference, the following files will be added:

- `input_text_encoded.npy`: token ids for the whole `input_text.txt`.
- `input_text_tokenized.json`: per-token decoded strings.
- `pos_tagged_tokens.json`: POS tag per token (plus sentinels).
- `begin_tokens.npy`, `end_tokens.npy`: indices of token boundaries in the original text (used by some analyses).
- `probabilities/<model>/<revision>/probabilities.npy`: flattened per-token log-probabilities as float16.
- `results/`: reduced arrays produced in stats-stage, split by experiment type:
  - `results/seeds/*.npy`: seed-vs-seed KL arrays, surprisals, etc.
  - `results/deduped/*.npy`: deduped Pythia results (code exists; not used in the paper).
  - `results/cross/*.npy`: cross comparisons (e.g., duped vs deduped where applicable).
- `output/*.csv` and `output/*.png`: dataframes and plots.

See `working_dir/sample1`, `working_dir/sample4`, `working_dir/sample_bert`, and `working_dir/icl_score` for concrete examples.

### 3) run-stage (inference)

Directory: `run-stage/`

Preprocessing (run before any runner):

- `tokenize_util.py <experiment_id>`: tokenize `input_text.txt` with Pythia tokenizer → saves `input_text_encoded.npy` and `input_text_tokenized.json`.
- `pos_tagger.py <experiment_id>`: NLTK POS-tags the raw text, aligns tags to HF tokens, and saves `pos_tagged_tokens.json`, `begin_tokens.npy`, `end_tokens.npy`.
- BERT variants: `tokenize_util_bert.py`, `pos_tagger_bert.py` for MultiBERTs tokenization/tag alignment.

Runners (produce `probabilities.npy`):

- `runner_seeds.py <experiment_id>`: Pythia seed-based runs. Reads `seeds_config.json` and writes one `probabilities.npy` per `(model, revision, seed)`.
- `runner.py <experiment_id>`: Pythia deduped runs. Reads `deduped_config.json`. Provided for completeness (not used in the paper).
- `runner_bert.py <experiment_id>`: MultiBERTs seed runs. Reads `seeds_config.json` and writes one `probabilities.npy` per `(model, revision, seed)` using BERT tokenization.

Custom experiment runners:

- ICL: `pythia_icl.py <experiment_id>` and `alternate_icl.py <experiment_id>` compute in-context learning (ICL) scores across steps/seeds from `seeds_config.json`. They write CSVs under `working_dir/<experiment_id>/output` (e.g., `alternate_icl.csv`).
- BLiMP: `blimp_task.py <experiment_id>` scores BLiMP minimal pairs for each seed/model/step, writes `blimp_distributions.csv` and `blimp_seed_kl.csv` to `output/`.

Quickstart (Pythia seeds):

```bash
python run-stage/tokenize_util.py sample4
python run-stage/pos_tagger.py sample4
python run-stage/runner_seeds.py sample4
```

Quickstart (MultiBERTs):

```bash
python run-stage/tokenize_util_bert.py sample_bert
python run-stage/pos_tagger_bert.py sample_bert
python run-stage/runner_bert.py sample_bert
```

One-shot helpers (Linux): `execute.sh` (deduped) and `execute_seeds.sh` (seeds) chain the core steps. Inspect them for the exact sequence.

Notes:
- Models and revisions are pulled from the Hugging Face Hub and cached locally. GPU is used if available.
- Logits are windowed with overlap to handle long sequences; outputs are concatenated and truncated so array length matches the number of next-token targets.

### 4) stats-stage (reduce probabilities to metrics)

Directory: `stats-stage/`

Given the per-token log-probabilities, these scripts write reduced arrays to `results/`:

- Pairwise model-to-model KL (per-token arrays):
  - Pythia seeds: `seeds_kl_torch.py <experiment_id>`
  - Pythia deduped: `deduped_kl_torch.py <experiment_id> <from_idx> <to_idx>` (not in paper) 
  - MultiBERTs: `bert_kl_torch.py <experiment_id> <seed_index>`

- KL vs reference distributions (uniform/unigram):
  - Pythia seeds: `seeds_kl_uni.py <experiment_id> <from_idx> <to_idx>` → uses `uniform_dist.npy` and `unigram_dist.npy`
  - Pythia deduped: `deduped_kl_uni.py <experiment_id> <from_idx> <to_idx>` (not in paper)
  - MultiBERTs: `bert_kl_uni.py <experiment_id>` → uses `uniform_dist_bert.npy` and `unigram_dist_bert.npy`

- Token-level surprisal (negative log-prob of the correct next token):
  - Pythia seeds: `seeds_stats.py <experiment_id>`
  - Pythia deduped: `deduped_stats.py <experiment_id>` (not in paper)
  - MultiBERTs: `bert_stats.py <experiment_id>`

- Cross comparisons:
  - `duped_vs_deduped_kl.py <experiment_id>`: compares deduped vs non-deduped models (where available)

Outputs are primarily arrays saved under `results/seeds` or `results/deduped` with names like:

- `...-seed{S}-kl.npy`: KL values for a fixed reference `(model, revision, seed)` vs all other `(model, revision, seed)` combinations; dims are `[num_comparisons, num_tokens]`.
- `...-seed{S}-uni.npy`: two rows for `[KL(uniform), KL(unigram)]`, each over tokens.
- `...-seed{S}-surprisal.npy`: per-token surprisal for a given `(model, revision, seed)`.

### 5) dataframes (CSV builders)

Directory: `dataframes/`

These scripts aggregate KL/surprisal arrays, token frequencies, and POS tags into CSVs under `output/`:

- `seeds_frequency_dataframe.py`: per-token KL paired with token frequency, POS of current/previous token, and surprisal at each step/seed.
- `seeds_uni_dataframe.py`: per-step averages of KL to uniform and unigram.
- `seeds_token_divergence_dataframe.py`: per-token average KL across seed pairs at the final step.
- `seeds_begin_end_dataframe.py`: begin/end token subsets vs overall averages for KL and surprisal.
- `seeds_pos_current_dataframe.py`: averages stratified by current-token POS and coarse groups (nouns/verbs); also correlates with surprisal.
- `seeds_pos_context_dataframe.py`: averages stratified by context-token POS categories.

Deduped (not in paper, but code provided):

- `deduped_frequency_dataframe.py`, `deduped_pos_current_dataframe.py`, `deduped_pos_context_dataframe.py`, `deduped_uni_dataframe.py` mirror the seeds versions.

Utility:

- `dataframes/utils.py`: helpers to read configs, POS tag sets (`common/pos_tags.json`), tokenized text and strip model/step names for plotting.

### 6) paper-plots (figure scripts)

Directory: `paper-plots/`

Each script reads CSVs from `working_dir/<experiment_id>/output`, renders a styled plot (see `paper-plots/utils.py`), and saves a `.png` beside the data.

- `binned_plot.py`: KL or cross-entropy binned by token frequency and by final-step surprisal.
- `cross_entropy_plot.py`: cross-entropy by POS categories; also supports grouped categories.
- `learning_rates.py`: nominal learning rate schedules by model size (reference figure).
- `linear_regression_plot.py`: plots linear regression coefficients relating frequency/POS to convergence.
- `multiberts.py`: expected convergence vs step for MultiBERTs across POS variants; supports legend trimming for clarity.
- `new_icl_plotter.py`: plot ICL score vs step from `alternate_icl.csv`.
- `rev2rev_pos_plot.py`: expected convergence vs step by POS (rev-to-rev comparisons).
- `surprisal_dataframe.py`: builds `seeds_surprisal_by_token.csv` for plots that join KL with surprisal.
- `token_divergence.py`: prints tokens with lowest/middle/highest average KL.
- `uni_plot.py`: expected KL vs step against uniform/unigram baselines.
- `variance_plot.py`: tokenwise variance/expected convergence curve across steps and model sizes.

Example invocations:

```bash
python dataframes/seeds_frequency_dataframe.py sample4
python paper-plots/binned_plot.py sample4
python paper-plots/variance_plot.py sample4
python run-stage/pythia_icl.py sample4 && python paper-plots/new_icl_plotter.py sample4
python run-stage/blimp_task.py sample4 && python paper-plots/blimp.py sample4
```

### Environment and dependencies

- Python 3.10+, PyTorch, Hugging Face `transformers` and `datasets`, `numpy`, `pandas`, `seaborn`, `matplotlib`, `nltk`, `unidecode`.
- NLTK resources are downloaded on first run in the POS taggers, but you can pre-download: `punkt`, `averaged_perceptron_tagger_eng`.
- A CUDA-capable GPU is recommended; the scripts fall back to CPU where possible. Probability arrays are stored as float16 to save disk space.

### Tips and gotchas

- Ensure your `seeds_config.json` and/or `deduped_config.json` list the exact HF model ids and revision names (e.g., `"EleutherAI/pythia-14m-seed1"` with revisions like `"step128"`).
- For BERT runs, text snippets are split on `[SEP][CLS]` inside `runner_bert.py` before scoring; ensure your `input_text.txt` uses that separator or adapt as needed.
- `deduped_*` scripts are included for reproducibility but were not used in the paper’s final results.

### End-to-end examples

Pythia seeds (minimal):

```bash
python run-stage/tokenize_util.py sample4
python run-stage/pos_tagger.py sample4
python run-stage/runner_seeds.py sample4
python stats-stage/seeds_kl_torch.py sample4
python stats-stage/seeds_kl_uni.py sample4 0 9999  # optional, slices model list
python stats-stage/seeds_stats.py sample4
python dataframes/seeds_frequency_dataframe.py sample4
python paper-plots/binned_plot.py sample4
```

MultiBERTs seeds:

```bash
python run-stage/tokenize_util_bert.py sample_bert
python run-stage/pos_tagger_bert.py sample_bert
python run-stage/runner_bert.py sample_bert
python stats-stage/bert_kl_torch.py sample_bert 0
python stats-stage/bert_kl_uni.py sample_bert
python stats-stage/bert_stats.py sample_bert
```

ICL and BLiMP (optional):

```bash
python run-stage/pythia_icl.py sample4 && python paper-plots/new_icl_plotter.py sample4
python run-stage/blimp_task.py sample4 && python paper-plots/blimp.py sample4
```

If anything is unclear, browse the scripts referenced above—they are short and designed to be read.