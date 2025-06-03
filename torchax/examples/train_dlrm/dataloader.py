import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
from typing import List, Dict, Iterator


VOCAB_SIZES = [
    40000000,
    39060,
    17295,
    7424,
    20265,
    3,
    7122,
    1543,
    63,
    40000000,
    3067956,
    405282,
    10,
    2209,
    11938,
    155,
    4,
    976,
    14,
    40000000,
    40000000,
    40000000,
    590152,
    12973,
    108,
    36,
]
MULTI_HOT_SIZES = [
    3,
    2,
    1,
    2,
    6,
    1,
    1,
    1,
    1,
    7,
    3,
    8,
    1,
    6,
    9,
    5,
    1,
    1,
    1,
    12,
    100,
    27,
    10,
    3,
    1,
    1,
]

class DummyCriteoDataset(Dataset):
    """
    A PyTorch Dataset that generates a dummy Criteo-like dataset in memory.
    This is equivalent to the `get_dummy_batch` and `_get_cached_dummy_dataset`
    functionality in the original code.
    """
    def __init__(
        self,
        num_samples: int,
        num_dense_features: int,
        vocab_sizes: List[int],
        multi_hot_sizes: List[int],
    ):
        super().__init__()
        self.num_samples = num_samples
        self.num_dense_features = num_dense_features
        self.vocab_sizes = vocab_sizes
        self.multi_hot_sizes = multi_hot_sizes
        self.num_sparse_features = len(vocab_sizes)

        # Generate all data at once and store in memory
        self.labels = torch.randint(0, 2, (self.num_samples,), dtype=torch.long)
        self.dense_features = torch.rand(self.num_samples, self.num_dense_features, dtype=torch.float32)

        self.sparse_features = {}
        for i in range(self.num_sparse_features):
            # Note: PyTorch embedding layers expect Long tensors (int64)
            self.sparse_features[str(i)] = torch.randint(
                low=0,
                high=self.vocab_sizes[i],
                size=(self.num_samples, self.multi_hot_sizes[i]),
                dtype=torch.long,
            )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a single data sample as a dictionary of tensors.
        """
        sparse_feats_sample = {
            key: val[idx] for key, val in self.sparse_features.items()
        }
        
        return {
            "clicked": self.labels[idx],
            "dense_features": self.dense_features[idx],
            "sparse_features": sparse_feats_sample,
        }


