# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

molformer4molalkit is a Python package that wraps IBM's MolFormer (a transformer-based molecular representation model) for molecular property prediction. It integrates with `mgktools` (via `Dataset` and evaluation utilities) and provides CLI entry points for cross-validation and hyperparameter optimization via Optuna.

## Installation

```bash
pip install -e .
```

Requires Python >= 3.10. Key runtime dependencies (not listed in pyproject.toml but required): `torch`, `pytorch_lightning`, `fast_transformers`, `apex`, `transformers`, `optuna`, `rdkit`, `mgktools`, `tap` (typed-argument-parser), `regex`, `sklearn`.

## CLI Entry Points

```bash
# Cross-validation or external test evaluation
molformer_cv --data_path <csv> --smiles_columns <col> --targets_columns <col> --task_type regression --save_dir <dir> ...

# Hyperparameter optimization with Optuna
molformer_optuna --data_path <csv> --smiles_columns <col> --targets_columns <col> --task_type regression --save_dir <dir> --n_trials 100 ...
```

Both are defined in `molformer/run.py` and use `tap.Tap` argument classes from `molformer/args.py`. Run with `--help` for all options.

## Architecture

- **`molformer/molformer.py`** — Core module. Contains:
  - `LightningModule` (pl.LightningModule): The transformer model with rotary attention, token embedding, mean-pooling, and a 2-layer MLP head (`Net`) with skip connections. Uses `apex.optimizers.FusedLAMB` optimizer. Loads from pretrained IBM MolFormer checkpoints.
  - `MolFormer`: High-level wrapper that manages ensemble training (`fit_molalkit`), prediction (`predict_value`), and uncertainty estimation (`predict_uncertainty`). Handles data loading and collation with the custom tokenizer.

- **`molformer/args.py`** — `TrainArgs` and `OptunaArgs` (extends `TrainArgs`) using `tap.Tap`. Handles data loading into `mgktools.data.data.Dataset` objects during `process_args()`. SMILES are canonicalized via RDKit on load.

- **`molformer/evaluator.py`** — `Evaluator` class orchestrating cross-validation (kFold, Monte-Carlo) and external test set evaluation. Outputs prediction CSVs and metrics CSVs to `save_dir`.

- **`molformer/tokenizer.py`** — `MolTranBertTokenizer` extending HuggingFace `BertTokenizer` with a SMILES-specific regex tokenizer. Vocabulary defined in `bert_vocab.txt`.

- **`molformer/rotate_attention/`** — Rotary positional encoding attention implementation (rotary embeddings + linear attention via `fast_transformers`).

- **`molformer/checkpoints/`** — Auto-discovers `.ckpt` files at runtime. Pretrained checkpoints must be placed here (downloaded from IBM MolFormer).

## Key Design Patterns

- Data flows through `mgktools.data.data.Dataset` objects — the package does not define its own dataset splitting; it delegates to `mgktools`.
- The model supports ensemble training: multiple models are trained with different seeds and predictions are averaged.
- Task types: `regression` (L1 loss) and `binary` classification (BCE loss with sigmoid at inference).
- NaN target values are masked during both training loss computation and metric evaluation.
- Optuna stores trials in a SQLite database (`optuna.db`) in the save directory, supporting resumable optimization.
