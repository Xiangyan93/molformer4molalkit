"""Integration tests for molformer_cv and molformer_optuna CLI functions.

These tests run the real CLI on a small fake CSV dataset with real checkpoints.
They require torch, pytorch_lightning, fast_transformers, apex, rdkit, mgktools, etc.
"""

import os
import csv
import shutil
import tempfile
import pytest

# Skip entire module if torch is not available
torch = pytest.importorskip("torch")

CWD = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CWD)
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "molformer", "checkpoints")
DEFAULT_CKPT = os.path.join(CHECKPOINT_DIR, "N-Step-Checkpoint_3_30000.ckpt")

# Skip if checkpoint not present
pytestmark = pytest.mark.skipif(
    not os.path.exists(DEFAULT_CKPT),
    reason="Pretrained checkpoint not found"
)

# A minimal set of SMILES and regression targets
_SMILES = [
    "CCO",
    "CCCO",
    "CC(=O)O",
    "c1ccccc1",
    "CC(C)O",
    "CCN",
    "CC=O",
    "CCCC",
    "CC(=O)N",
    "c1ccc(O)cc1",
]
_TARGETS = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

_BINARY_TARGETS = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

# Multi-task targets with missing values (NaN)
_MULTI_TARGETS = [
    [1.0, 2.0],
    [2.0, None],  # missing value for task 1
    [3.0, 4.0],
    [4.0, None],  # missing value for task 1
    [5.0, 6.0],
    [6.0, 7.0],
    [7.0, None],  # missing value for task 1
    [8.0, 9.0],
    [9.0, 10.0],
    [10.0, 11.0],
]

_FEATURES = [
    [0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0],
    [1.1, 1.2], [1.3, 1.4], [1.5, 1.6], [1.7, 1.8], [1.9, 2.0],
]


def _write_csv(path, smiles, targets, target_col="target"):
    """Write a minimal CSV with smiles and target columns."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["smiles", target_col])
        for s, t in zip(smiles, targets):
            writer.writerow([s, t])


def _write_csv_with_features(path, smiles, targets, features):
    """Write a CSV with smiles, feature columns, and target."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["smiles", "feat1", "feat2", "target"])
        for s, feats, t in zip(smiles, features, targets):
            writer.writerow([s, feats[0], feats[1], t])


def _write_multitask_csv(path, smiles, multi_targets):
    """Write a CSV with smiles and multiple target columns (may contain None for missing)."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["smiles", "target1", "target2"])
        for s, targets in zip(smiles, multi_targets):
            # Convert None to empty string for CSV (will be read as NaN)
            row = [s] + [t if t is not None else "" for t in targets]
            writer.writerow(row)


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp(prefix="molformer_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def regression_csv(tmp_dir):
    path = os.path.join(tmp_dir, "data.csv")
    _write_csv(path, _SMILES, _TARGETS)
    return path


@pytest.fixture
def binary_csv(tmp_dir):
    path = os.path.join(tmp_dir, "data_bin.csv")
    _write_csv(path, _SMILES, _BINARY_TARGETS)
    return path


@pytest.fixture
def features_csv(tmp_dir):
    path = os.path.join(tmp_dir, "features_data.csv")
    _write_csv_with_features(path, _SMILES, _TARGETS, _FEATURES)
    return path


@pytest.fixture
def ext_test_csv(tmp_dir):
    """A separate small test set."""
    path = os.path.join(tmp_dir, "test.csv")
    _write_csv(path, _SMILES[:3], _TARGETS[:3])
    return path


@pytest.fixture
def multitask_csv(tmp_dir):
    """Multi-task CSV with missing values."""
    path = os.path.join(tmp_dir, "multitask.csv")
    _write_multitask_csv(path, _SMILES, _MULTI_TARGETS)
    return path


class TestMolformerCVIntegration:
    """Integration tests for molformer_cv with real model and small data."""

    def test_kfold_regression(self, tmp_dir, regression_csv):
        from molformer.run import molformer_cv

        save_dir = os.path.join(tmp_dir, "kfold_out")
        molformer_cv([
            "--data_path", regression_csv,
            "--smiles_columns", "smiles",
            "--targets_columns", "target",
            "--task_type", "regression",
            "--metric", "rmse",
            "--cross_validation", "kFold",
            "--n_splits", "2",
            "--num_folds", "1",
            "--epochs", "1",
            "--batch_size", "5",
            "--ensemble_size", "1",
            "--n_jobs", "1",
            "--save_dir", save_dir,
        ])

        assert os.path.isdir(save_dir)
        assert os.path.exists(os.path.join(save_dir, "kFold_metrics.csv"))

    def test_monte_carlo_regression(self, tmp_dir, regression_csv):
        from molformer.run import molformer_cv

        save_dir = os.path.join(tmp_dir, "mc_out")
        molformer_cv([
            "--data_path", regression_csv,
            "--smiles_columns", "smiles",
            "--targets_columns", "target",
            "--task_type", "regression",
            "--metric", "rmse",
            "--cross_validation", "Monte-Carlo",
            "--split_type", "random",
            "--split_sizes", "0.8", "0.2",
            "--num_folds", "1",
            "--epochs", "1",
            "--batch_size", "5",
            "--ensemble_size", "1",
            "--n_jobs", "1",
            "--save_dir", save_dir,
        ])

        assert os.path.exists(os.path.join(save_dir, "Monte-Carlo_metrics.csv"))

    def test_external_test_regression(self, tmp_dir, regression_csv, ext_test_csv):
        from molformer.run import molformer_cv

        save_dir = os.path.join(tmp_dir, "ext_out")
        molformer_cv([
            "--data_path", regression_csv,
            "--smiles_columns", "smiles",
            "--targets_columns", "target",
            "--task_type", "regression",
            "--metric", "rmse",
            "--epochs", "1",
            "--batch_size", "5",
            "--ensemble_size", "1",
            "--n_jobs", "1",
            "--save_dir", save_dir,
            "--separate_test_path", ext_test_csv,
        ])

        assert os.path.exists(os.path.join(save_dir, "test_ext_prediction.csv"))
        assert os.path.exists(os.path.join(save_dir, "test_ext_metrics.csv"))

    def test_kfold_regression_with_features(self, tmp_dir, features_csv):
        from molformer.run import molformer_cv

        save_dir = os.path.join(tmp_dir, "kfold_features_out")
        molformer_cv([
            "--data_path", features_csv,
            "--smiles_columns", "smiles",
            "--targets_columns", "target",
            "--features_columns", "feat1", "feat2",
            "--task_type", "regression",
            "--metric", "rmse",
            "--cross_validation", "kFold",
            "--n_splits", "2",
            "--num_folds", "1",
            "--epochs", "1",
            "--batch_size", "5",
            "--ensemble_size", "1",
            "--n_jobs", "1",
            "--save_dir", save_dir,
        ])

        assert os.path.isdir(save_dir)
        assert os.path.exists(os.path.join(save_dir, "kFold_metrics.csv"))

    def test_kfold_regression_with_features_generators(self, tmp_dir, regression_csv):
        from molformer.run import molformer_cv

        save_dir = os.path.join(tmp_dir, "kfold_fg_out")
        molformer_cv([
            "--data_path", regression_csv,
            "--smiles_columns", "smiles",
            "--targets_columns", "target",
            "--features_generators_name", "rdkit_2d_normalized",
            "--task_type", "regression",
            "--metric", "rmse",
            "--cross_validation", "kFold",
            "--n_splits", "2",
            "--num_folds", "1",
            "--epochs", "1",
            "--batch_size", "5",
            "--ensemble_size", "1",
            "--n_jobs", "1",
            "--save_dir", save_dir,
        ])

        assert os.path.isdir(save_dir)
        assert os.path.exists(os.path.join(save_dir, "kFold_metrics.csv"))

    def test_binary_kfold(self, tmp_dir, binary_csv):
        from molformer.run import molformer_cv

        save_dir = os.path.join(tmp_dir, "bin_out")
        molformer_cv([
            "--data_path", binary_csv,
            "--smiles_columns", "smiles",
            "--targets_columns", "target",
            "--task_type", "binary",
            "--metric", "roc_auc",
            "--cross_validation", "kFold",
            "--n_splits", "2",
            "--num_folds", "1",
            "--epochs", "1",
            "--batch_size", "5",
            "--ensemble_size", "1",
            "--n_jobs", "1",
            "--save_dir", save_dir,
        ])

        assert os.path.exists(os.path.join(save_dir, "kFold_metrics.csv"))

    def test_multitask_with_missing_values(self, tmp_dir, multitask_csv):
        """Test multi-task learning with missing values in targets."""
        import pandas as pd
        from molformer.run import molformer_cv

        save_dir = os.path.join(tmp_dir, "multitask_out")
        molformer_cv([
            "--data_path", multitask_csv,
            "--smiles_columns", "smiles",
            "--targets_columns", "target1", "target2",
            "--task_type", "regression",
            "--metric", "rmse",
            "--cross_validation", "kFold",
            "--n_splits", "2",
            "--num_folds", "1",
            "--epochs", "1",
            "--batch_size", "5",
            "--ensemble_size", "1",
            "--n_jobs", "1",
            "--save_dir", save_dir,
        ])

        # Verify metrics file exists and has correct structure
        metrics_path = os.path.join(save_dir, "kFold_metrics.csv")
        assert os.path.exists(metrics_path)

        df_metrics = pd.read_csv(metrics_path)
        # Should have n_samples column for weighted averaging
        assert "n_samples" in df_metrics.columns
        # Task 0 (target1) should have more samples than Task 1 (target2)
        task0_samples = df_metrics[df_metrics["no_targets_columns"] == 0]["n_samples"].sum()
        task1_samples = df_metrics[df_metrics["no_targets_columns"] == 1]["n_samples"].sum()
        assert task0_samples > task1_samples, "Task 0 should have more samples than Task 1"


class TestMolformerOptunaIntegration:
    """Integration tests for molformer_optuna with real model and small data."""

    def test_optuna_single_trial_regression(self, tmp_dir, regression_csv):
        from molformer.run import molformer_optuna

        save_dir = os.path.join(tmp_dir, "optuna_out")
        molformer_optuna([
            "--data_path", regression_csv,
            "--smiles_columns", "smiles",
            "--targets_columns", "target",
            "--task_type", "regression",
            "--metric", "rmse",
            "--cross_validation", "kFold",
            "--n_splits", "2",
            "--num_folds", "1",
            "--n_jobs", "1",
            "--save_dir", save_dir,
            "--n_trials", "1",
        ])

        assert os.path.exists(os.path.join(save_dir, "optuna.db"))
        assert os.path.isdir(os.path.join(save_dir, "trial-0"))

    def test_optuna_resume_skips_done(self, tmp_dir, regression_csv):
        """Run 1 trial, then call again with n_trials=1 — should not run more."""
        from molformer.run import molformer_optuna

        save_dir = os.path.join(tmp_dir, "optuna_resume")
        common_args = [
            "--data_path", regression_csv,
            "--smiles_columns", "smiles",
            "--targets_columns", "target",
            "--task_type", "regression",
            "--metric", "rmse",
            "--cross_validation", "kFold",
            "--n_splits", "2",
            "--num_folds", "1",
            "--n_jobs", "1",
            "--save_dir", save_dir,
            "--n_trials", "1",
        ]
        molformer_optuna(common_args)
        # Second call: n_trials=1, 1 already done => 0 to run
        molformer_optuna(common_args)

        # Still only trial-0 directory (no trial-1)
        assert os.path.isdir(os.path.join(save_dir, "trial-0"))
        assert not os.path.isdir(os.path.join(save_dir, "trial-1"))
