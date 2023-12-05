import torch
import random
import numpy as np


class Shift():
    def __init__(self, n_trans, max_offset=0, H=128, W=128):
        self.n_trans = n_trans
        self.max_offset=max_offset
        self.shifts_row = random.sample(list(np.concatenate([-1 * np.arange(1, H), np.arange(1, W)])), n_trans)
        self.shifts_col = random.sample(list(np.concatenate([-1 * np.arange(1, H), np.arange(1, W)])), n_trans)
    def apply(self, x):
        H, W = x.shape[-2], x.shape[-1]
        assert self.n_trans <= H - 1 and self.n_trans <= W - 1, 'n_shifts should less than {}'.format(H - 1)

        x = torch.cat([x if self.n_trans == 0 else torch.roll(x, shifts=[sx, sy], dims=[-2, -1]).type_as(x) for sx, sy in
                       zip(self.shifts_row, self.shifts_col)], dim=0)
        return x
