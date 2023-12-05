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


num_splits = 2
batch_size = num_splits
test_angles = 1024   # angles used for testing
test_I0 = 1e4   # I0 for testing
angles = 1024   # angles used for training
I0 = 1e4      # I0  for training
method = 'RAN2I'

input_dir = Path("data/test_split")
weights_path = Path("weights/weights_{}_{}_{}.torch".format(method, angles, I0))
output_dir = Path("result/{}_{}_{}".format(method, angles, I0))
os.makedirs(output_dir, exist_ok=True)

datasets = [np_dataset(input_dir / f"{j}/*.npy") for j in range(num_splits)]
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

# np.save(os.path.join(output_dir, 'output_{}_{}_{}.npy'.format(method, angles, I0)), output)

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

average_psnr = np.mean(PSNR)
average_ssim = np.mean(SSIM)

print(f"average PSNR:  {average_psnr:5.2f}")
print(f"average SSIM:  {average_ssim:5.2f}")

std_psnr = np.std(PSNR)
std_ssim = np.std(SSIM)

print(f"std PSNR:  {std_psnr:5.2f}")
print(f"std SSIM:  {std_ssim:5.2f}")


# display and save image
plt.imshow(output[0], cmap='gray')
plt.imsave(os.path.join(output_dir, 'output_{}_{}.png'.format(test_angles, test_I0)), output[0], cmap='gray')
plt.show()
