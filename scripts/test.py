import matplotlib.pyplot as plt
import numpy as np
import torch
from dataset import np_dataset, Noise2InverseDataset
from torch.utils.data import DataLoader
from pathlib import Path
from models.dncnn import DnCNN
from tqdm import tqdm
import os
import glob
import skimage.metrics as skm
from physics import split


def eval_RAN2I(n_split, test_angles, test_I0, train_angles, train_I0, geom):
    n_split = n_split
    batch_size = n_split
    test_angles = test_angles   # angles used for testing
    test_I0 = test_I0   # I0 for testing
    train_angles = train_angles   # angles used for training
    train_I0 = train_I0      # I0  for training
    method = 'RAN2I'
    split(n_split=n_split, angles=test_angles, I0=test_I0, mode="test", geom=geom)

    input_dir = Path("data/test_split")
    weights_path = Path("weights/weights_{}_{}_{}.torch".format(method, train_angles, train_I0))
    output_dir = Path("result/{}_{}_{}".format(method, train_angles, train_I0))
    os.makedirs(output_dir, exist_ok=True)

    datasets = [np_dataset(input_dir / f"{j}/*.npy") for j in range(n_split)]
    ds = Noise2InverseDataset(*datasets)
    dl = DataLoader(ds, batch_size, shuffle=False)

    net = DnCNN()
    state = torch.load(weights_path)
    net.load_state_dict(state["state_dict"])
    net = net.cuda()

    output = []
    net.eval()
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dl)):
            inp, _ = batch
            inp = inp.cuda()
            out = net(inp)
            out = out.mean(dim=0)
            out_np = out.detach().cpu().numpy().squeeze()
            output.append(out_np)

    clean_file = sorted(glob.glob(os.path.join('data', 'test_data', '*.npy')))
    # calculate PSNR, SSIM
    PSNR = []
    SSIM = []
    for i in range(len(clean_file)):
        clean = np.load(clean_file[i])
        out = output[i]

        data_range = clean.max() - clean.min()
        psnr = skm.peak_signal_noise_ratio(clean, out, data_range=data_range)
        ssim = skm.structural_similarity(clean, out, data_range=data_range)

        PSNR.append(psnr)
        SSIM.append(ssim)

    RAN2I_psnr = np.mean(PSNR)
    RAN2I_ssim = np.mean(SSIM)

    print(f"RAN2I PSNR:  {RAN2I_psnr:5.2f}")
    print(f"RAN2I SSIM:  {RAN2I_ssim:5.2f}")

    # save image
    # plt.imshow(output[0], cmap='gray')
    # plt.show()
    plt.imsave(os.path.join(output_dir, 'output_{}_{}.png'.format(test_angles, test_I0)), output[0], cmap='gray')

    return output[0]
