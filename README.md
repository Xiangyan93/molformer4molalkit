# molformer4molalkit

A customized version of [IBM MolFormer](https://github.com/IBM/molformer) for molecular active learning. This package wraps the MolFormer transformer model for molecular property prediction and is designed to be used with [MolALKit](https://github.com/RekerLab/MolALKit).

## Pretrained Checkpoints

This package requires pretrained MolFormer checkpoint files to function. Download them from the [IBM MolFormer repository](https://github.com/IBM/molformer) and place them in the following directory:

```
molformer/checkpoints/
```

For example:

```
molformer/checkpoints/N-Step-Checkpoint_3_30000.ckpt
```

All `.ckpt` files in this directory are automatically discovered at runtime. If no `--pretrained_path` is specified, the default checkpoint `N-Step-Checkpoint_3_30000.ckpt` is used.

## CLI Usage

Two command-line tools are provided:

### `molformer_cv` -- Cross-validation and external test evaluation

```bash
molformer_cv \
    --data_path data.csv \
    --smiles_columns smiles \
    --targets_columns target \
    --task_type regression \
    --metric rmse \
    --cross_validation kFold \
    --n_splits 5 \
    --num_folds 1 \
    --epochs 50 \
    --save_dir results/
```

Supported cross-validation modes: `kFold`, `Monte-Carlo`, `no` (use with `--separate_test_path`).

To evaluate on an external test set instead of cross-validation:

```bash
molformer_cv \
    --data_path train.csv \
    --separate_test_path test.csv \
    --smiles_columns smiles \
    --targets_columns target \
    --task_type regression \
    --metric rmse \
    --save_dir results/
```

### `molformer_optuna` -- Hyperparameter optimization

```bash
molformer_optuna \
    --data_path data.csv \
    --smiles_columns smiles \
    --targets_columns target \
    --task_type regression \
    --metric rmse \
    --cross_validation kFold \
    --n_splits 5 \
    --save_dir optuna_results/ \
    --n_trials 100
```

Optuna trials are stored in a SQLite database (`optuna.db`) in the save directory, so optimization can be resumed by re-running the same command.

Run either command with `--help` for all available options.
