"""Unit tests for the Evaluator class."""

import numpy as np
import pandas as pd
import pytest

from molformer.evaluator import Evaluator


class TestWeightedMean:
    """Tests for the _weighted_mean static method."""

    def test_weighted_mean_single_row(self):
        """Single row should return the value itself."""
        df = pd.DataFrame({
            "metric": ["rmse"],
            "no_targets_columns": [0],
            "value": [1.5],
            "n_samples": [100]
        })
        result = Evaluator._weighted_mean(df)
        assert result == 1.5

    def test_weighted_mean_equal_samples(self):
        """Equal samples should give same result as simple mean."""
        df = pd.DataFrame({
            "metric": ["rmse", "rmse"],
            "no_targets_columns": [0, 1],
            "value": [1.0, 2.0],
            "n_samples": [100, 100]
        })
        result = Evaluator._weighted_mean(df)
        assert result == 1.5  # (1.0 + 2.0) / 2

    def test_weighted_mean_unequal_samples(self):
        """Unequal samples should weight by sample count."""
        df = pd.DataFrame({
            "metric": ["rmse", "rmse"],
            "no_targets_columns": [0, 1],
            "value": [1.0, 2.0],
            "n_samples": [100, 50]
        })
        # Weighted: (1.0 * 100 + 2.0 * 50) / (100 + 50) = 200 / 150 = 1.333...
        result = Evaluator._weighted_mean(df)
        expected = (1.0 * 100 + 2.0 * 50) / 150
        assert abs(result - expected) < 1e-10

    def test_weighted_mean_with_nan_value(self):
        """NaN values should be excluded from calculation."""
        df = pd.DataFrame({
            "metric": ["rmse", "rmse", "rmse"],
            "no_targets_columns": [0, 1, 2],
            "value": [1.0, np.nan, 3.0],
            "n_samples": [100, 50, 100]
        })
        # Only task 0 and 2 count: (1.0 * 100 + 3.0 * 100) / 200 = 2.0
        result = Evaluator._weighted_mean(df)
        assert result == 2.0

    def test_weighted_mean_all_nan(self):
        """All NaN values should return NaN."""
        df = pd.DataFrame({
            "metric": ["rmse", "rmse"],
            "no_targets_columns": [0, 1],
            "value": [np.nan, np.nan],
            "n_samples": [100, 50]
        })
        result = Evaluator._weighted_mean(df)
        assert np.isnan(result)

    def test_weighted_mean_zero_samples(self):
        """Zero total samples should return NaN."""
        df = pd.DataFrame({
            "metric": ["rmse", "rmse"],
            "no_targets_columns": [0, 1],
            "value": [1.0, 2.0],
            "n_samples": [0, 0]
        })
        result = Evaluator._weighted_mean(df)
        assert np.isnan(result)

    def test_weighted_mean_no_n_samples_column(self):
        """Without n_samples column, fallback to simple mean."""
        df = pd.DataFrame({
            "metric": ["rmse", "rmse"],
            "no_targets_columns": [0, 1],
            "value": [1.0, 3.0]
        })
        result = Evaluator._weighted_mean(df)
        assert result == 2.0  # simple mean

    def test_weighted_mean_multiple_folds(self):
        """Multiple folds should aggregate correctly."""
        df = pd.DataFrame({
            "metric": ["rmse", "rmse", "rmse", "rmse"],
            "no_targets_columns": [0, 1, 0, 1],
            "value": [1.0, 2.0, 1.5, 2.5],
            "n_samples": [50, 25, 50, 25],  # task 0 has 100 total, task 1 has 50 total
            "seed": [0, 0, 1, 1]
        })
        # Weighted: (1.0*50 + 2.0*25 + 1.5*50 + 2.5*25) / (50+25+50+25)
        # = (50 + 50 + 75 + 62.5) / 150 = 237.5 / 150 = 1.583...
        result = Evaluator._weighted_mean(df)
        expected = (1.0*50 + 2.0*25 + 1.5*50 + 2.5*25) / 150
        assert abs(result - expected) < 1e-10
