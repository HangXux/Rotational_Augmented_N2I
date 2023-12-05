import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from itertools import combinations


class np_dataset(Dataset):

    def __init__(self, glob):
        super(np_dataset, self).__init__()
        glob = Path(glob)
        self.glob = glob
        self.paths = sorted(Path(glob.parent).glob(glob.name))

    def __getitem__(self, i):
        img = np.load(str(self.paths[i]))
        if img.ndim == 2:
            img = img[None, ...]
        img = torch.from_numpy(img).float()

        return img

    def __len__(self):
        return len(self.paths)

class Noise2InverseDataset(Dataset):
    """Documentation for Noise2InverseDataset
    reference: https://github.com/ahendriksen/noise2inverse"""
    def __init__(self, *datasets, strategy="X:1"):
        super(Noise2InverseDataset, self).__init__()

        self.datasets = datasets
        max_len = max(len(ds) for ds in datasets)
        min_len = min(len(ds) for ds in datasets)

        assert min_len == max_len

        assert strategy in ["X:1", "1:X"]
        self.strategy = strategy

        if strategy == "X:1":
            num_input = self.num_splits - 1
        else:
            num_input = 1

        # For num_splits=2, we have
        # input_idxs =  [{0}, {1}]
        # target_idxs = [{1}, {0}]
        split_idxs = set(range(self.num_splits))
        self.input_idxs = list(combinations(split_idxs, num_input))
        self.target_idxs = [split_idxs - set(idxs) for idxs in self.input_idxs]

    @property
    def num_splits(self):
        return len(self.datasets)

    @property
    def num_slices(self):
        return len(self.datasets[0])

    def __getitem__(self, i):
        num_splits = self.num_splits
        slice_idx = i // num_splits
        split_idx = i % num_splits

        input_idxs = self.input_idxs[split_idx]
        target_idxs = self.target_idxs[split_idx]

        slices = [ds[slice_idx] for ds in self.datasets]
        inputs = [slices[j] for j in input_idxs]
        targets = [slices[j] for j in target_idxs]

        inp = torch.mean(torch.stack(inputs), dim=0)
        tgt = torch.mean(torch.stack(targets), dim=0)

        return inp, tgt

    def __len__(self):
        return self.num_splits * self.num_slices
