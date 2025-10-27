# Data Anatomy: Mixture Estimation and Evaluation

This repo provides two complementary methods to estimate category mixtures of language model training data, plus a standardized evaluator to compare runs against ground-truth specs.

- Label-shift (classifier + prior correction)
- Min-K%++ (membership inference over local samples)
- Evaluator with accuracy metric and plots

## Quick Start

1) Install dependencies (GPU recommended for HF models):

```
pip install -U torch transformers scikit-learn numpy pandas tqdm matplotlib pyyaml datasets accelerate seaborn tabulate
# Optional for SlimPajama .jsonl.zst sampling
pip install zstandard
```

2) Prepare local category samples (optional if you already have them):

```
python fetch_category_samples.py \
  --slimpajama_root /path/to/SlimPajama-627B-DC/train \
  --n_per_category 5000 \
  --out_dir data_samples
```

This writes JSONL files per category under `data_samples/` with expected names:
`commoncrawl.jsonl, c4.jsonl, github.jsonl, wikipedia.jsonl, books.jsonl, arxiv.jsonl, stackexchange.jsonl`.

## Methods

### Label-Shift (Classifier + Prior Correction)

Trains a domain classifier over local samples, generates text from a target LM, predicts class probabilities for the generations, and solves for mixture priors.

Example (6-class; merge CommonCrawl + C4 → Web):

```
python baseline_method/src/labelshift/run_labelshift.py \
  --local_samples_dir data_samples \
  --merge_web \
  --classifier distilbert \
  --target_model allenai/OLMo-1B \
  --hf_revision step20000-tokens84B \
  --num_prompts 300 \
  --max_new_tokens 512 \
  --output_dir out \
  --run_name labelshift_olmo1b
```

Outputs (in `out/labelshift_olmo1b/`):
- `summary.json` (priors with mean/CI, confusion matrix, averaged probabilities)
- `summary.csv`
- Diagnostic plots: `confusion_matrix.png`, `priors.png`, `pbar_vs_ctpi.png`

Notes:
- Use `--use_cached_generations path/to/gens.jsonl` to skip HF generation (expects `{"text": ...}` per line).
- Switch classifier to TF-IDF with `--classifier tfidf` for faster, lighter runs.

### Min-K%++ (Membership Inference over Local Samples)

Estimates a per-category membership rate via GMM over Min-K% standardized token log-prob scores, then normalizes across categories to obtain the global mixture.

Example (6-class):

```
python baseline_method/src/labelshift/run_minkpp_mix.py \
  --local_samples_dir data_samples \
  --merge_web \
  --target_model EleutherAI/pythia-2.8b \
  --mink_ratio 0.1 \
  --max_tokens 128 \
  --output_dir out \
  --run_name minkpp_mix
```

Outputs (in `out/minkpp_mix/`):
- `summary.json` (key field: `global_mixture_over_sampled_pool`)
- `per_category.csv` (per-category membership rates and diagnostics)
- `global_mixture.csv`

Tips:
- Use `--half` or `--int8` to reduce memory footprint.

## Evaluation

Use `benchmark_evaluation.py` to score one or more runs against a ground-truth spec under `bench/specs/*.yaml`.

Key features:
- Auto-aligns 6-class vs 7-class schemes (merges CommonCrawl + C4 into Web if needed).
- Reads priors from `summary.json` for label-shift or `global_mixture_over_sampled_pool` for Min-K%++.
- Adds accuracy derived from absolute error: `Accuracy = 1 − 0.5 × L1` (in [0,1]).
- Optional within-tolerance accuracy: fraction of categories with `|e_c − g_c| ≤ τ`.

### Single Method

```
python benchmark_evaluation.py \
  --results_dir out/minkpp_mix \
  --ground_truth bench/specs/olmo1b.yaml \
  --tol 0.02 \
  --output_dir benchmark_output
```

Prints metrics and writes plots/report to `benchmark_output/` (omit `--no_plots/--no_report` to get both).

### Compare Multiple Methods

```
python benchmark_evaluation.py \
  --compare out/labelshift_olmo1b out/minkpp_mix \
  --ground_truth bench/specs/olmo1b.yaml \
  --tol 0.02 \
  --output_dir benchmark_output
```

### Accuracy Definition (for papers)

We report “overlap accuracy,” the complement of total variation distance. With ground-truth `g` and estimate `e`:

```
Acc = 1 − 0.5 × Σ_c |e_c − g_c|
```

This equals 1 when `e=g` and decreases linearly as mass is misplaced. We optionally report `Acc_tol(τ) = mean_c[ |e_c − g_c| ≤ τ ]`.

## Ground-Truth Specs

Ground-truth mixtures live in YAML files under `bench/specs/`, e.g.:
- `bench/specs/olmo1b.yaml`
- `bench/specs/pythia.yaml`
- `bench/specs/starcoder.yaml`

Each `category_weights` map is normalized by the evaluator before scoring.

## Repro Tips

- GPU strongly recommended for HF generation and Min-K%++ scoring.
- Set `HF_HOME` and log in to HuggingFace if needed to access models.
- For large local datasets (SlimPajama), ensure `zstandard` is installed; the fetcher can sample from `.jsonl.zst` directly.

