from scripts.test import eval_RAN2I
import matplotlib.pyplot as plt
import numpy as np
from physics import FBP
import argparse


def main(args):

    gt = np.load(args.data_path)
    fbp = FBP(input=gt, angles=args.test_angles, I0=args.test_I0, geom=args.geometry)
    ran2i = eval_RAN2I(n_split=args.n_split, test_angles=args.test_angles, test_I0=args.test_I0,
                       train_angles=args.train_angles, train_I0=args.train_I0, geom=args.geometry)

    plt.figure(figsize=[16, 6], layout="tight")
    plt.subplot(131), plt.imshow(gt, cmap="gray"), plt.title("ground truth"), plt.axis("off")
    plt.subplot(132), plt.imshow(fbp, cmap="gray"), plt.title("FBP"), plt.axis("off")
    plt.subplot(133), plt.imshow(ran2i, cmap="gray"), plt.title("RAN2I"), plt.axis("off")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default="./data/test_data/L506_202_full.npy")
    parser.add_argument("--geometry", type=str, default="parallel")
    parser.add_argument("--n_split", type=int, default=2,
                        help="number of splits")
    parser.add_argument("--test_angles", type=int, default=1024,
                        help="sampling angles for testing data")
    parser.add_argument("--test_I0", type=float, default=1e4,
                        help="incident photon counts for testing")
    parser.add_argument("--train_angles", type=int, default=1024,
                        help="sampling angels used in training")
    parser.add_argument("--train_I0", type=float, default=1e4,
                        help="incident photon counts in training")

    args = parser.parse_args()
    main(args)
