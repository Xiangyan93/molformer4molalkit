from tap import Tap
from typing import List, Literal, Optional
import os
import copy
import pandas as pd
from sklearn.model_selection import KFold
from rdkit import Chem
from mgktools.data.data import Dataset
from mgktools.evaluators.cross_validation import Metric
from molformer.checkpoints import AVAILABLE_CHECKPOINTS
CWD = os.path.dirname(__file__)


class TrainArgs(Tap):
    save_dir: str = None
    """Directory where model checkpoints will be saved."""
    pretrained_path: str = None
    """Path to a pretrained model checkpoint."""
    n_jobs: int = 1
    """The cpu numbers used for parallel computing."""
    data_path: str = None
    """The Path of input data CSV file."""
    smiles_columns: List[str] = None
    """List of names of the columns containing SMILES strings.
    By default, uses the first :code:`number_of_molecules` columns."""
    targets_columns: List[str] = None
    """
    Name of the columns containing target values.
    """
    task_type: Literal["regression", "binary", "multi-class"]
    """Type of task."""
    cross_validation: Literal["kFold", "leave-one-out", "Monte-Carlo", "no"] = "no"
    """The way to split data for cross-validation."""
    n_splits: int = None
    """The number of fold for kFold CV."""
    split_type: Literal["random", "scaffold_order", "scaffold_random", "stratified"] = None
    """Method of splitting the data into train/test sets."""
    split_sizes: List[float] = None
    """Split proportions for train/test sets."""
    num_folds: int = 1
    """Number of folds when performing cross validation."""
    seed: int = 0
    """Random seed."""
    metric: Metric = None
    """metric"""
    extra_metrics: List[Metric] = []
    """Metrics"""
    separate_test_path: str = None
    """Path to separate test set, optional."""
    separate_val_path: str = None
    """Path to separate validation set, optional."""
    n_head: int = 12
    """Number of attention heads."""
    n_layer: int = 12
    """Number of transformer layers."""
    n_embd: int = 768
    """Dimensionality of the embeddings."""
    d_dropout: float = 0.1
    """Dropout rate for the attention layers."""
    dropout: float = 0.1
    """Dropout rate for the feed-forward layers."""
    learning_rate: float = 3e-5
    """Learning rate for the optimizer."""
    num_feats: int = 32
    """Number of features."""
    ensemble_size: int = 1
    """Number of models in the ensemble."""
    epochs: int = 50
    """Number of training epochs."""
    batch_size: int = 128
    """Batch size for training."""
    weight_decay: float = 0.0
    """Weight decay (L2 regularization) factor."""

    @property
    def metrics(self) -> List[Metric]:
        return [self.metric] + self.extra_metrics
    
    def get_df(self, file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path)
        for col in self.smiles_columns:
            df[col] = df[col].apply(lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x)))
        return df

    def process_args(self) -> None:
        if self.pretrained_path is None:
            self.pretrained_path = os.path.join(CWD, 'checkpoints', 'N-Step-Checkpoint_3_30000.ckpt')
        self.dataset = Dataset.from_df(
            df=self.get_df(self.data_path),
            smiles_columns=self.smiles_columns,
            features_columns=None,
            targets_columns=self.targets_columns,
            n_jobs=self.n_jobs,
        )
        self.dataset.set_status(graph_kernel_type='no',
                                features_generators=None,
                                features_combination=None)
        if self.separate_test_path is not None:
            self.dataset_test = Dataset.from_df(
                df=self.get_df(self.separate_test_path),
                smiles_columns=self.smiles_columns,
                features_columns=None,
                targets_columns=self.targets_columns,
                n_jobs=self.n_jobs,
            )
            self.dataset_test.set_status(graph_kernel_type='no',
                                        features_generators=None,
                                        features_combination=None)
        if self.separate_val_path is not None:
            self.dataset_val = Dataset.from_df(
                df=self.get_df(self.separate_val_path),
                smiles_columns=self.smiles_columns,
                features_columns=None,
                targets_columns=self.targets_columns,
                n_jobs=self.n_jobs,
            )
            self.dataset_val.set_status(graph_kernel_type='no',
                                        features_generators=None,
                                        features_combination=None)
            self.dataset_train_val = copy.deepcopy(self.dataset)
            self.dataset_train_val.data = self.dataset.data + self.dataset_val.data


class OptunaArgs(TrainArgs):
    n_trials: int = 100
    """Number of Optuna trials to perform."""
