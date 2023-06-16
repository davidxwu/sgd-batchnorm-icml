import argparse
import os
import time
from argparse import Namespace

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from scipy.spatial import ConvexHull
from sklearn.datasets import make_blobs
from sklearn.random_projection import GaussianRandomProjection
from torch import nn
from torch.utils.data import (DataLoader, Dataset, RandomSampler,
                              SequentialSampler)
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor

parser = argparse.ArgumentParser(description="PyTorch Toy Data Training")
parser.add_argument(
    "--model",
    default="2l_lnn",
    type=str,
    help="model, one of bn_1l_lnn, 1l_lnn, 2l_lnn, 3l_lnn, 2l_fc, 3l_fc",
)
parser.add_argument(
    "--charlie_dataset",
    default=False,
    action="store_true",
    help="whether to use charlie dataset",
)
parser.add_argument(
    "--toy_dataset",
    default=False,
    action="store_true",
    help="whether to use toy dataset",
)
parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
parser.add_argument("--bsize", default=2, type=int, help="minibatch size")
parser.add_argument("--num_datapoints", default=2, type=int, help="size of dataset")
parser.add_argument(
    "--sampling",
    default="RR",
    type=str,
    help="sampling type, one of RR, SS, IGM, or SGD",
)
parser.add_argument(
    "--seed", default=-1, type=int, help="seed for randomness (nondet if not included)"
)
parser.add_argument("--width", default=512, type=int, help="hidden layer width")
parser.add_argument("--epoch", default=100, type=int, help="number of training epochs")
parser.add_argument(
    "--bn", dest="bn", default=False, action="store_true", help="use batchnorm layers"
)
parser.add_argument(
    "--freeze_gamma", default=False, action="store_true", help="use bn but freeze gamma"
)
parser.add_argument(
    "--dn",
    dest="dn",
    default=False,
    action="store_true",
    help="data normalization. [0,1] to [-1,1]",
)
parser.add_argument("--momen", default=0.0, type=float, help="momentum param of SGD")
parser.add_argument(
    "--rn", default=1, type=int, help="run number"
)  # just to keep log files separate
parser.add_argument("--gpu", default=0, type=int, help="designate GPU number")
parser.add_argument("--resume", default=0, type=int, help="read and resume")
parser.add_argument("--suffix", default="", type=str, help="suffix for log file")
args: Namespace = parser.parse_args()


writer = SummaryWriter(
    comment=f"_toy_data_shuffling_{args.sampling}_lr_{args.lr}_epoch_{args.epoch}_model_{args.model}_num_datapoints_{args.num_datapoints}_bsize_{args.bsize}_bn_{args.bn}_freeze_{args.freeze_gamma}_seed_{args.seed}_charlie_dataset_{args.charlie_dataset}_{args.suffix}"
)
transformer1 = GaussianRandomProjection(n_components=2)
transformer2 = GaussianRandomProjection(n_components=2)
transformer3 = GaussianRandomProjection(n_components=2)  # fix 3 projections from start

activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


# Define models
class LNN1(nn.Module):  # aka the linear model
    def __init__(self, input_size=2):
        super(LNN1, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(input_size, 1, bias=False)
        self.bn_1 = nn.BatchNorm1d(
            input_size, affine=not args.freeze_gamma, eps=0, track_running_stats=False
        )
        self.bn_1_no_affine = nn.BatchNorm1d(
            input_size, affine=False, eps=0, track_running_stats=False
        )

        self.linear_1.weight.data = torch.Tensor([[-1, 1]])
        # nn.init.constant_(linear_1, )
        if not args.bn:
            self.linear_relu_stack = nn.Sequential(self.linear_1)
            self.feature_stack = nn.Sequential(nn.Identity())
        else:
            self.linear_relu_stack = nn.Sequential(self.bn_1, self.linear_1)
            self.feature_stack = nn.Sequential(self.bn_1_no_affine)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def get_features(self, x):
        x = self.flatten(x)
        return self.feature_stack(x)


class BNLNN1(nn.Module):  # aka BN(WX)
    def __init__(self, input_size=2):
        super(BNLNN1, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(input_size, 1, bias=False)
        self.bn_1 = nn.BatchNorm1d(1, track_running_stats=False)
        self.bn_1_no_affine = nn.BatchNorm1d(1, affine=False, track_running_stats=False)

        if not args.bn:
            self.linear_relu_stack = nn.Sequential(self.linear_1)
            self.feature_stack = nn.Sequential(nn.Identity())
        else:
            self.linear_relu_stack = nn.Sequential(self.linear_1, self.bn_1)
            self.feature_stack = nn.Sequential(self.linear_1, self.bn_1_no_affine)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def get_features(self, x):
        x = self.flatten(x)
        return self.feature_stack(x)


class LNN2(nn.Module):
    def __init__(self, input_size=2):
        super(LNN2, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(args.width, 1, bias=False)
        self.linear_2 = nn.Linear(input_size, args.width, bias=False)

        self.bn_1 = nn.BatchNorm1d(
            args.width, affine=not args.freeze_gamma, track_running_stats=False
        )
        self.bn_1_no_affine = nn.BatchNorm1d(
            args.width, affine=False, track_running_stats=False
        )

        if not args.bn:
            self.linear_relu_stack = nn.Sequential(self.linear_2, self.linear_1)
            self.feature_stack = nn.Sequential(
                self.linear_2,
            )
        else:
            self.linear_relu_stack = nn.Sequential(
                self.linear_2, self.bn_1, self.linear_1
            )
            self.feature_stack = nn.Sequential(
                self.linear_2,
                self.bn_1_no_affine,
            )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def get_features(self, x):
        x = self.flatten(x)
        return self.feature_stack(x)


class LNN3(nn.Module):
    def __init__(self, input_size=2):
        super(LNN3, self).__init__()
        self.flatten = nn.Flatten()

        self.linear_1 = nn.Linear(args.width, 1, bias=False)
        self.linear_2 = nn.Linear(args.width, args.width, bias=False)
        self.linear_3 = nn.Linear(input_size, args.width, bias=False)

        self.bn_1 = nn.BatchNorm1d(args.width, track_running_stats=False)
        self.bn_1_no_affine = nn.BatchNorm1d(
            args.width, affine=False, track_running_stats=False
        )
        self.bn_2 = nn.BatchNorm1d(args.width, track_running_stats=False)
        if not args.bn:
            self.linear_relu_stack = nn.Sequential(
                self.linear_3, self.linear_2, self.linear_1
            )
            self.feature_stack = nn.Sequential(
                self.linear_3,
                self.linear_2,
            )
        else:
            self.linear_relu_stack = nn.Sequential(
                self.linear_3, self.bn_2, self.linear_2, self.bn_1, self.linear_1
            )
            self.feature_stack = nn.Sequential(
                self.linear_3,
                self.bn_2,
                self.linear_2,
                self.bn_1_no_affine,
            )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def get_features(self, x):
        x = self.flatten(x)
        return self.feature_stack(x)


class RandomSamplerSS(RandomSampler):
    def __init__(self, train_dataset):
        super().__init__(train_dataset)
        self.epoch = 1
        self.permutation = None

    def __iter__(self):
        n = len(self.data_source)
        if self.epoch == 1:
            generator = torch.Generator()
            if args.seed == -1:
                seed = int(torch.empty((), dtype=torch.int64).random_().item())
            else:
                seed = args.seed
            generator.manual_seed(seed)
            writer.add_text("seed", str(seed), 1)
            self.permutation = torch.randperm(n, generator=generator).tolist()
            yield from self.permutation
        else:
            yield from self.permutation
        self.epoch = self.epoch + 1


class ToyDataset(Dataset):
    def __init__(self, m):
        B = np.hstack(
            (
                np.linspace(-2.2, -1.8, m).reshape(m, 1),
                np.linspace(1.8, 2.2, m).reshape(m, 1),
            )
        )

        # B = np.hstack((np.linspace(-2+1/m, -2+2/m, m).reshape(m, 1), np.linspace(2+1/m, 2+2/m, m).reshape(m, 1)))
        D = -B
        # A = np.array([[-1, 1], [-0.5, 0.5]])
        A = np.array([[-3, 1.5], [1, -0.5]])
        C = -A
        repeats = 1
        misclass = np.tile(np.array([[300, 110], [-300, -110]]), (repeats, 1))
        # print(misclass.shape)
        # misclass = np.array([[3, 2.5], [-3, -2.5]])

        X = np.vstack((A, B, C, D, misclass))
        Y = np.hstack(([1, 0], np.ones(m), [0, 1], np.zeros(m), [1, 0] * repeats))
        # print(X, Y)
        self.X = torch.Tensor(X).float().T
        self.y = torch.Tensor(Y).float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[:, idx], self.y[idx]


class BlobsDataset(Dataset):
    def __init__(
        self,
        num_samples=12,
        num_features=2,
        centers=2,
        transform=None,
        target_transform=None,
        random_state=None,
    ):
        self.X, self.y = make_blobs(
            n_samples=num_samples,
            n_features=num_features,
            centers=centers,
            random_state=args.seed if random_state is None else random_state,
        )
        self.transform = transform
        self.target_transform = target_transform
        if self.transform is not None:
            self.X = self.transform(self.X).float()
        if self.target_transform is not None:
            self.y = self.target_transform(self.y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[:, idx], self.y[idx]


class CharlieDataset(Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.X1 = torch.Tensor(
            [[1, 0], [0, 1], [0.01, -0.01], [-0.01, 0.01], [0.01, -0.01], [-0.01, 0.01]]
        )
        self.X2 = torch.Tensor(
            [[0.01, -0.01], [-0.01, 0.01], [1, 0], [0, 1], [-0.01, 0.01], [0.01, -0.01]]
        )
        self.X3 = torch.Tensor(
            [[0.01, -0.01], [-0.01, 0.01], [0.01, -0.01], [-0.01, 0.01], [1, 0], [0, 1]]
        )
        self.y = torch.Tensor([1, 1, 1, 0, 0, 0])
        self.X = torch.hstack((self.X1, self.X2, self.X3))
        self.transform = transform
        self.target_transform = target_transform
        if self.transform is not None:
            self.X = self.transform(self.X).float()
        if self.target_transform is not None:
            self.y = self.target_transform(self.y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[:, idx], self.y[idx]


def plot_convex_hulls(
    X,
    y,
    pos_color="b",
    neg_color="r",
    pos_label="Positive",
    neg_label="Negative",
    ax=None,
):
    positive_labels = y == 1
    positive_features = X[positive_labels, :]
    negative_features = X[~positive_labels, :]
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(
        positive_features[:, 0],
        positive_features[:, 1],
        "+",
        color=pos_color,
        markersize=10,
        mew=3,
        alpha=0.8,
        label=pos_label,
    )
    ax.plot(
        negative_features[:, 0],
        negative_features[:, 1],
        "o",
        color=neg_color,
        alpha=0.8,
        label=neg_label,
    )
    positive_hull = ConvexHull(positive_features)
    negative_hull = ConvexHull(negative_features)
    for simplex in positive_hull.simplices:
        ax.plot(
            positive_features[simplex, 0],
            positive_features[simplex, 1],
            color=pos_color,
            linestyle="--",
            alpha=0.5,
        )
    for simplex in negative_hull.simplices:
        ax.plot(
            negative_features[simplex, 0],
            negative_features[simplex, 1],
            color=neg_color,
            linestyle="--",
            alpha=0.5,
        )
    ax.legend()


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss += loss_fn(torch.sigmoid(pred).squeeze(), y.float())

        # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # if batch % 50 == 0:
    loss, current = loss.item(), batch * len(X)
    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def calcloss(
    train_eval_dataloader,
    test_eval_dataloader,
    model,
    loss_fn,
    epochidx,
    savefilename,
    save=True,
):
    model.eval()
    size = len(train_eval_dataloader.dataset)
    train_loss, train_correct = 0, 0
    combined_features = []
    combined_y = []

    with torch.no_grad():
        for X, y in train_eval_dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(torch.sigmoid(pred).squeeze(), y.float())
            train_loss += loss.item() * 2
            pred_binary = (pred >= 0).squeeze()
            train_correct += (pred_binary == y).type(torch.float).sum().item()

            # visualize data in 2d
            if epochidx % 200 == 0:
                learned_features = model.get_features(X).cpu().numpy()
                combined_features.append(learned_features)
                combined_y.append(y.cpu().numpy())

        # print(learned_SS_features)
        if epochidx % 200 == 0:
            learned_SS_features = np.vstack(combined_features)
            y_SS = np.hstack(combined_y)
            print(learned_SS_features)
            d = learned_SS_features.shape[-1]
            fig, ax = plt.subplots()
            plot_convex_hulls(
                learned_SS_features,
                y_SS,
                pos_color="purple",
                neg_color="green",
                pos_label="Positive (SS)",
                neg_label="Negative (SS)",
                ax=ax,
            )
            ax.set_title(f"2d visualization of SS features at epoch {epochidx}")

            W = model.linear_1.weight
            # print(W)
            #     W = param
            plt.axline(
                (0, 0),
                slope=-W[0, 0].item() / W[0, 1].item(),
                color="green",
                linestyle=":",
                label=r"$(v_{\pi}^*)^\top x = 0$",
            )

        # if epochidx % 1000 == 0:
        #
    train_loss /= size
    train_correct /= size
    print(
        f"Training Error: \n Accuracy: {(100 * train_correct):>0.1f}%, Avg loss: {train_loss:>8f} \n"
    )
    writer.add_scalar("SS risk/train", train_loss, epochidx)
    writer.add_scalar("SS acc/train", train_correct, epochidx)

    size = len(test_eval_dataloader.dataset)
    test_loss, test_correct = 0, 0
    with torch.no_grad():
        for X, y in test_eval_dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(torch.sigmoid(pred).squeeze(), y.float())
            test_loss += loss.item() * test_eval_batch_size
            pred_binary = (pred >= 0).squeeze()
            test_correct += (pred_binary == y).type(torch.float).sum().item()

            if epochidx % 200 == 0:
                learned_GD_features = model.get_features(X).cpu().numpy()
                print(learned_GD_features)
                plot_convex_hulls(
                    learned_GD_features,
                    y.cpu().numpy(),
                    pos_color="blue",
                    neg_color="red",
                    pos_label="Positive (GD)",
                    neg_label="Negative (GD)",
                    ax=ax,
                )
                writer.add_figure("2d visualization", fig, epochidx)

    test_loss /= size
    test_correct /= size
    print(
        f"Test Error: \n Accuracy: {(100 * test_correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )
    writer.add_scalar("GD risk/train", test_loss, epochidx)
    writer.add_scalar("GD acc/train", test_correct, epochidx)
    Gamma = None
    W = None
    # log weight norms every 100 epochs
    if epochidx % 100 == 0:
        for name, param in model.named_parameters():
            param_size = np.sqrt(
                torch.numel(param)
            )  # sqrt is the correct normalization here
            writer.add_scalar(
                f"{name} norm normalized by size", param.norm() / param_size, epochidx
            )
            writer.add_scalar(f"{name} norm unnormalized", param.norm(), epochidx)

    if save:
        f = open(savefilename, "a")
        f.write(
            str(epochidx)
            + " "
            + str(train_loss)
            + " "
            + str(train_correct)
            + " "
            + str(test_loss)
            + " "
            + str(test_correct)
        )
        f.write("\n")
        f.close()


if __name__ == "__main__":
    start_time = time.time()
    os.makedirs("./results/", exist_ok=True)
    os.makedirs("./model/", exist_ok=True)

    if args.resume <= 0:
        filename = "toy_data"
        for key, value in vars(args).items():
            if key == "gpu" or key == "resume":
                continue
            else:
                filename = filename + "_" + str(value)
        print("Results will be stored in ./results/" + filename + ".txt")
    else:
        oldfilename = "toy_data"
        filename = "toy_data"
        for key, value in vars(args).items():
            if key == "gpu" or key == "resume":
                continue
            elif key == "epoch":
                oldfilename = oldfilename + "_" + str(args.resume)
                filename = filename + "_" + str(args.epoch + args.resume)
            else:
                oldfilename = oldfilename + "_" + str(value)
                filename = filename + "_" + str(value)
        print("Loading checkpoint stored in ./model/" + oldfilename + ".pth and resume")
        print("Results will be stored in ./results/" + filename + ".txt")

        checkpoint = torch.load("./model/" + oldfilename + ".pth")
        fold = open("./results/" + oldfilename + ".txt", "r")
        fnew = open("./results/" + filename + ".txt", "a")
        lines = fold.readlines()
        for i in range(len(lines) - 1):
            fnew.write(lines[i])
        fold.close()
        fnew.close()

    if args.dn:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    else:
        transform = transforms.ToTensor()

    if args.charlie_dataset:
        training_data = CharlieDataset()
        test_data = CharlieDataset()
        input_size = 6
    elif args.toy_dataset:
        training_data = ToyDataset(m=args.num_datapoints)
        test_data = ToyDataset(m=args.num_datapoints)
        input_size = 2
    else:
        training_data = BlobsDataset(
            num_samples=args.num_datapoints,
            num_features=2,
            centers=[[0, 0], [3, 3]],
            random_state=args.seed + 1,
            transform=ToTensor(),
        )

        test_data = BlobsDataset(
            num_samples=args.num_datapoints,
            num_features=2,
            centers=[[0, 0], [3, 3]],
            random_state=args.seed + 2,
            transform=ToTensor(),
        )
        input_size = 2

    batch_size = args.bsize
    train_eval_batch_size = len(training_data)
    test_eval_batch_size = len(test_data)

    # Create data loaders.
    custom_sampler = None
    sampling = args.sampling
    if sampling == "SGD":
        custom_sampler = RandomSampler(data_source=training_data, replacement=True)
    elif sampling == "SS":
        custom_sampler = RandomSamplerSS(training_data)
    elif sampling == "RR":
        custom_sampler = RandomSampler(data_source=training_data)
    elif sampling == "IGM":
        custom_sampler = SequentialSampler(data_source=training_data)
    else:
        raise ValueError("--sampling argument should be one of SGD, SS or RR.")

    train_dataloader = DataLoader(
        training_data, batch_size=batch_size, sampler=custom_sampler, num_workers=2
    )
    train_plot_dataloader = DataLoader(
        training_data, batch_size=train_eval_batch_size, num_workers=2
    )
    test_plot_dataloader = DataLoader(
        test_data, batch_size=test_eval_batch_size, num_workers=2
    )

    # Get cpu or gpu device for training.
    device = "cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    # Define model.
    if args.model == "1l_lnn":
        model = LNN1(input_size=input_size).to(device)
    elif args.model == "bn_1l_lnn":
        model = BNLNN1(input_size=input_size).to(device)
    elif args.model == "2l_lnn":
        model = LNN2(input_size=input_size).to(device)
    elif args.model == "3l_lnn":
        model = LNN3().to(device)
    elif args.model == "2l_fc":
        model = FC2().to(device)
    elif args.model == "3l_fc":
        model = FC3().to(device)
    else:
        raise ValueError(
            "--model argument should be one of 1l_lnn, 2l_lnn, 3l_lnn, 2l_fc, 3l_fc."
        )
    print(model)

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momen)

    if args.resume <= 0:
        calcloss(
            train_dataloader,
            train_plot_dataloader,
            model,
            loss_fn,
            0,
            "./results/" + filename + ".txt",
        )

        for t in range(args.epoch):
            print(f"Epoch {t + 1}\n-------------------------------")
            train(train_dataloader, model, loss_fn, optimizer)
            calcloss(
                train_dataloader,
                train_plot_dataloader,
                model,
                loss_fn,
                t + 1,
                "./results/" + filename + ".txt",
            )
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if args.sampling == "SS":
            custom_sampler.permutation = checkpoint["SS_perm"]
            custom_sampler.epoch = args.resume

        calcloss(
            train_dataloader,
            train_plot_dataloader,
            model,
            loss_fn,
            args.resume,
            "./results/" + filename + ".txt",
            False,
        )
        for t in range(args.resume, args.resume + args.epoch):
            print(f"Epoch {t + 1}\n-------------------------------")
            train(train_dataloader, model, loss_fn, optimizer)
            calcloss(
                train_plot_dataloader,
                test_plot_dataloader,
                model,
                loss_fn,
                t + 1,
                "./results/" + filename + ".txt",
            )

    print("Done!")

    sfn = "./model/" + filename + ".pth"

    if sampling == "SS":
        state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "SS_perm": custom_sampler.permutation,
        }
    else:
        state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
    torch.save(state, sfn)
    print("Saved PyTorch Model State to " + sfn)
    writer.add_hparams(vars(args), {})
    writer.flush()
    writer.close()
    f = open("./results/" + filename + ".txt", "a")
    f.write("Execution time: %s seconds" % (time.time() - start_time))
    f.close()
