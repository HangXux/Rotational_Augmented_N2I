from scripts.train import train_RAN2I
import argparse


def main(args):
    n_split = args.n_split
    train_angles = args.train_angles
    train_I0 = args.train_I0
    geom = args.geometry
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    n_trans = args.n_trans

    train_RAN2I(n_split, train_angles, train_I0, geom, epochs, batch_size, lr, n_trans, random_rotate=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_split", type=int, default=2, help="number of splits")
    parser.add_argument("--train_angles", type=int, default=256, help="number of sampling angles")
    parser.add_argument("--train_I0", type=float, default=1e5, help="number of incident photon counts")
    parser.add_argument("--geometry", type=str, default="parallel")
    parser.add_argument("--epochs", type=int, default=100, help="total epochs for training")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size of training")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate of training")
    parser.add_argument("--n_trans", type=int, default=4, help="number of transformations in each group")

    args = parser.parse_args()
    main(args)
