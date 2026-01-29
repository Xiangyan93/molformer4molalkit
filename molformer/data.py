#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import List, Optional, Tuple, Union
import numpy as np
from torch.utils.data import Dataset


class MolFormerDataset(Dataset):
    """Dataset class for molecular data."""

    def __init__(self, smiles_list: List[str],
                 targets: List[Union[int, float]],
                 features: Optional[np.ndarray] = None):
        self.smiles_list = smiles_list
        self.targets = targets
        self.features = features
        assert len(smiles_list) == len(targets), "Length of SMILES list and targets must be the same."
        if features is not None:
            assert len(smiles_list) == len(features), "Length of SMILES list and features must be the same."

    def __len__(self) -> int:
        return len(self.smiles_list)

    def __getitem__(self, index: int):
        if self.features is not None:
            return self.smiles_list[index], self.features[index], self.targets[index]
        return self.smiles_list[index], self.targets[index]
