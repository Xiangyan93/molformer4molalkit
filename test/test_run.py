"""Unit tests for molformer_cv and molformer_optuna CLI functions."""

import sys
import types
import pytest
from unittest.mock import MagicMock, patch


def _make_fake_args(cls_name="TrainArgs", **overrides):
    """Return a mock args object with default attribute values."""
    args = MagicMock()
    args.save_dir = "/tmp/test_save"
    args.pretrained_path = "/fake/checkpoint.ckpt"
    args.task_type = "regression"
    args.targets_columns = ["target"]
    args.smiles_columns = ["smiles"]
    args.n_head = 12
    args.n_layer = 12
    args.n_embd = 768
    args.d_dropout = 0.1
    args.dropout = 0.1
    args.learning_rate = 3e-5
    args.num_feats = 32
    args.ensemble_size = 1
    args.epochs = 50
    args.batch_size = 128
    args.weight_decay = 0.0
    args.n_jobs = 1
    args.seed = 0
    args.n_features = 0
    args.metrics = ["rmse"]
    args.cross_validation = "kFold"
    args.n_splits = 5
    args.split_type = "random"
    args.split_sizes = [0.8, 0.2]
    args.num_folds = 1
    args.dataset = MagicMock()
    args.dataset_test = MagicMock()
    args.dataset_val = MagicMock()
    args.dataset_train_val = MagicMock()
    args.separate_test_path = None
    args.separate_val_path = None
    if cls_name == "OptunaArgs":
        args.n_trials = 100
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


# ---------------------------------------------------------------------------
# Setup: mock heavy dependencies so molformer.run can be imported
# ---------------------------------------------------------------------------

# Create mock modules for missing heavy deps before importing molformer.run
_MOCK_MODULES = [
    "torch", "torch.nn", "torch.nn.functional", "torch.utils",
    "torch.utils.data", "torch.optim",
    "pytorch_lightning", "pytorch_lightning.callbacks",
    "fast_transformers", "fast_transformers.builders",
    "fast_transformers.builders.base", "fast_transformers.builders.transformer_builders",
    "fast_transformers.builders.attention_builders",
    "fast_transformers.masking", "fast_transformers.feature_maps",
    "fast_transformers.transformers", "fast_transformers.attention",
    "fast_transformers.events",
    "apex", "apex.optimizers",
    "transformers",
    "regex",
]

_saved = {}
for mod_name in _MOCK_MODULES:
    _saved[mod_name] = sys.modules.get(mod_name)
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

# scipy's is_torch_array does issubclass(x, torch.Tensor) which fails if
# torch.Tensor is a MagicMock. Provide a real class to avoid TypeError.
class _FakeTensor:
    pass

sys.modules["torch"].Tensor = _FakeTensor

# Now we can import the module under test
# We also need to mock the actual classes used in run.py
_mock_MolFormer = MagicMock()
_mock_Evaluator = MagicMock()
_mock_TrainArgs = MagicMock()
_mock_OptunaArgs = MagicMock()

# Patch at module level before import
import molformer.run as run_module


def teardown_module():
    """Remove mock modules so integration tests can use real dependencies."""
    for mod_name in _MOCK_MODULES:
        if _saved[mod_name] is None:
            sys.modules.pop(mod_name, None)
        else:
            sys.modules[mod_name] = _saved[mod_name]
    # Also remove cached molformer submodules so they get re-imported with real deps
    for key in list(sys.modules):
        if key.startswith("molformer"):
            del sys.modules[key]

# ---------------------------------------------------------------------------
# molformer_cv tests
# ---------------------------------------------------------------------------


class TestMolformerCV:
    def setup_method(self):
        self._patchers = [
            patch.object(run_module, "TrainArgs"),
            patch.object(run_module, "MolFormer"),
            patch.object(run_module, "Evaluator"),
        ]
        self.MockTrainArgs, self.MockMolFormer, self.MockEvaluator = [
            p.start() for p in self._patchers
        ]

    def teardown_method(self):
        for p in self._patchers:
            p.stop()

    def test_cross_validation_path(self):
        """When separate_test_path is None, run_cross_validation is called."""
        args = _make_fake_args(separate_test_path=None)
        self.MockTrainArgs.return_value.parse_args.return_value = args

        run_module.molformer_cv()

        self.MockMolFormer.assert_called_once()
        self.MockEvaluator.assert_called_once()
        self.MockEvaluator.return_value.run_cross_validation.assert_called_once()
        self.MockEvaluator.return_value.run_external.assert_not_called()

    def test_external_test_path(self):
        """When separate_test_path is set, run_external is called."""
        args = _make_fake_args(separate_test_path="/fake/test.csv")
        self.MockTrainArgs.return_value.parse_args.return_value = args

        run_module.molformer_cv()

        self.MockEvaluator.return_value.run_external.assert_called_once_with(
            args.dataset_test
        )
        self.MockEvaluator.return_value.run_cross_validation.assert_not_called()

    def test_model_receives_correct_args(self):
        """MolFormer constructor receives key args from parsed arguments."""
        args = _make_fake_args()
        self.MockTrainArgs.return_value.parse_args.return_value = args

        run_module.molformer_cv()

        call_kwargs = self.MockMolFormer.call_args[1]
        assert call_kwargs["save_dir"] == args.save_dir
        assert call_kwargs["task_type"] == args.task_type
        assert call_kwargs["learning_rate"] == args.learning_rate
        assert call_kwargs["epochs"] == args.epochs


# ---------------------------------------------------------------------------
# molformer_optuna tests
# ---------------------------------------------------------------------------


class TestMolformerOptuna:
    def setup_method(self):
        self._patchers = [
            patch.object(run_module, "OptunaArgs"),
            patch.object(run_module, "optuna"),
        ]
        self.MockOptunaArgs, self.mock_optuna = [p.start() for p in self._patchers]

    def teardown_method(self):
        for p in self._patchers:
            p.stop()

    def _setup_study(self, n_trials=10, existing_trials=0):
        args = _make_fake_args(cls_name="OptunaArgs", n_trials=n_trials)
        self.MockOptunaArgs.return_value.parse_args.return_value = args

        mock_study = MagicMock()
        mock_study.trials = [MagicMock()] * existing_trials
        self.mock_optuna.create_study.return_value = mock_study
        return args, mock_study

    def test_creates_study_and_optimizes(self):
        """Verify optuna.create_study and study.optimize are called."""
        _, mock_study = self._setup_study(n_trials=10, existing_trials=0)

        run_module.molformer_optuna()

        self.mock_optuna.create_study.assert_called_once()
        mock_study.optimize.assert_called_once()
        call_kwargs = mock_study.optimize.call_args
        n = call_kwargs[1].get("n_trials") if call_kwargs[1] else call_kwargs[0][1]
        assert n == 10

    def test_skips_completed_trials(self):
        """When all trials done, optimize is not called."""
        _, mock_study = self._setup_study(n_trials=10, existing_trials=10)

        run_module.molformer_optuna()

        mock_study.optimize.assert_not_called()

    def test_partial_resume(self):
        """Only remaining trials are run."""
        _, mock_study = self._setup_study(n_trials=10, existing_trials=7)

        run_module.molformer_optuna()

        call_kwargs = mock_study.optimize.call_args
        n = call_kwargs[1].get("n_trials") if call_kwargs[1] else call_kwargs[0][1]
        assert n == 3

    def test_study_direction_regression(self):
        """Regression tasks use 'minimize' direction."""
        args, _ = self._setup_study(n_trials=1, existing_trials=0)
        args.task_type = "regression"

        run_module.molformer_optuna()

        call_kwargs = self.mock_optuna.create_study.call_args[1]
        assert call_kwargs["direction"] == "minimize"

    def test_study_direction_binary(self):
        """Binary tasks use 'maximize' direction."""
        args, _ = self._setup_study(n_trials=1, existing_trials=0)
        args.task_type = "binary"

        run_module.molformer_optuna()

        call_kwargs = self.mock_optuna.create_study.call_args[1]
        assert call_kwargs["direction"] == "maximize"
