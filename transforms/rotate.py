import torch
import numpy as np
import random
import kornia as dgm

class Rotate():
    def __init__(self, n_trans, random_rotate=True):
        self.n_trans = n_trans
        self.random_rotate = random_rotate
        if random_rotate:
            self.theta_list = random.sample(list(np.arange(1, 359)), n_trans)
        else:
            self.theta_list = np.arange(30, 360, int(360 / n_trans))
    def apply(self, x):
        return rotate_dgm(x, self.theta_list)

        # if random_rotate:
        #     theta_list = random.sample(list(np.arange(1, 359)), n_trans)
        # else:
        #     theta_list = np.arange(30, 360, int(360 / n_trans))


def rotate_dgm(data, theta_list):
# def rotate_dgm(data, n_trans=5, random_rotate=False):
    # if random_rotate:
    #     theta_list = random.sample(list(np.arange(1, 359)), n_trans)
    # else:
    #     theta_list = np.arange(30, 360, int(360 / n_trans))

    data = torch.cat([data if theta == 0 else dgm.geometry.transform.rotate(data, torch.Tensor([theta]).type_as(data))
                      for theta in theta_list], dim=0)
    return data