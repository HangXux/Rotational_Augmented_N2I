from pathlib import Path
from dataset import np_dataset, Noise2InverseDataset
import torch
from torch.utils.data import DataLoader
from models.dncnn import DnCNN
from transforms.rotate import Rotate
import numpy as np
import os
from physics import split


def train_RAN2I():
    # simulated settings
    n_split = 2  # number of split
    angles = 1024  # number of angles
    I0 = 1e4

    # split data
    split(n_split=n_split, angles=angles, I0=I0, mode='train')

    # training settings
    epochs = 100
    batch_size = 4
    output_dir = Path("weights")
    output_dir.mkdir(exist_ok=True)
    train_dir = Path("data/train_split")
    net = DnCNN().cuda()
    optimizer = torch.optim.Adam(net.parameters())
    criterion = torch.nn.MSELoss().cuda()

    # load datasets
    datasets = [np_dataset(train_dir / f"{j}/*.npy") for j in range(n_split)]
    train_ds = Noise2InverseDataset(*datasets)
    dl = DataLoader(train_ds, batch_size, shuffle=True)

    # training
    train_epochs = max(epochs // n_split, 1)
    save_loss_train = []

    for i in range(train_epochs):
        print("-----epoch {}-----".format(i+1))
        train_epoch_loss = 0
        train_step = 0
        for (inp, tgt) in dl:
            train_step += 1
            inp = inp.cuda()
            tgt = tgt.cuda()
            out = net(inp)  # output of the network
            loss_n2i = criterion(tgt, out)

            # rotate
            T = Rotate(n_trans=4, random_rotate=True)
            out_rotate = T.apply(out)  # rotate output
            tgt_rotate = T.apply(tgt)  # rotate target
            loss_rot = criterion(tgt_rotate, out_rotate)

            # joint loss
            loss = loss_n2i + loss_rot

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if train_step % 40 == 0:
                print("Step: " + str(train_step) + "  Loss: {}".format(loss.item()))

            train_epoch_loss += loss.item()

        print('-' * 20)

        train_epoch_loss /= train_step
        print(f'Epoch_loss: {train_epoch_loss:.4f}')
        save_loss_train.append(train_epoch_loss)

        # torch.save(
        #     {"epoch": int(i), "state_dict": net.state_dict(), "optimizer": optimizer.state_dict()},
        #     output_dir / f"weights_epoch_{i}.torch"
        # )

    # save loss
    np.save(os.path.join(output_dir, 'loss_train_RAN2I_{}_{}.npy'.format(angles, I0)), save_loss_train)

    # save weights
    torch.save(
        {"epoch": int(i), "state_dict": net.state_dict(), "optimizer": optimizer.state_dict()},
        output_dir / "weights_RAN2I_{}_{}.torch".format(angles, I0)
    )
