from argparse import Namespace

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import RandomSampler
import time
import torchvision
from torch.utils.data import RandomSampler
import argparse
import os
from sklearn.datasets import make_blobs, make_moons
from sklearn.model_selection import train_test_split
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import polytope
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet18, ResNet18_Weights, vgg16_bn, VGG16_BN_Weights

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--model', default='2l_lnn', type=str, help='model, one of bn_1l_lnn, 1l_lnn, 2l_lnn, 3l_lnn, 2l_fc, 3l_fc, resnet_18')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--bsize', default=128, type=int, help='minibatch size')
parser.add_argument('--sampling', default="RR", type=str, help='sampling type, one of RR, SS or SGD')
parser.add_argument('--width', default=512, type=int, help='hidden layer width')
parser.add_argument('--epoch', default=100, type=int, help='number of training epochs')
parser.add_argument('--bn', dest='bn', default=False, action='store_true', help='use batchnorm layers')
parser.add_argument('--ema', dest='ema', default=False, action='store_true', help='use EMA to estimate population stats')
parser.add_argument('--dn', dest='dn', default=False, action='store_true', help='data normalization. [0,1] to [-1,1]')
parser.add_argument('--momen', default=0.0, type=float, help='momentum param of SGD')
parser.add_argument('--rn', default=1, type=int, help='run number')  # just to keep log files separate
parser.add_argument('--gpu', default=0, type=int, help='designate GPU number')
parser.add_argument('--resume', default=0, type=int, help='read and resume')
parser.add_argument('--freeze_gamma', default=False, action='store_true', help='use bn but freeze gamma')
parser.add_argument('--suffix', default='', type=str, help='suffix for log file')
parser.add_argument('--num_classes', default=10, type=int, help='How many classes to use in CIFAR10')
parser.add_argument('--seed', default=-1, type=int, help='seed for randomness (nondet if not included)')
parser.add_argument('--eval_type', default='GD', type=str, help='Whether to use GD/SS risk to eval')
args: Namespace = parser.parse_args()

writer = SummaryWriter(comment=f'cifar10_shuffling_{args.sampling}_lr_{args.lr}_epoch_{args.epoch}_width_{args.width}_bsize_{args.bsize}_momen_{args.momen}_ema_{args.ema}_model_{args.model}_eval_type_{args.eval_type}_{args.suffix}')

# Define models
# class LNN1(nn.Module):  # aka the linear model
#     def __init__(self):
#         super(LNN1, self).__init__()
#         self.flatten = nn.Flatten()
#         self.linear_1 = nn.Linear(3 * 32 * 32, 10, bias=False)
#         self.bn_1 = nn.BatchNorm1d(3 * 32 * 32, track_running_stats=args.ema)
#         if not args.bn:
#             self.linear_relu_stack = nn.Sequential(
#                 self.linear_1
#             )
#         else:

#             self.linear_relu_stack = nn.Sequential(
#                 self.bn_1,
#                 self.linear_1
#             )

#     def forward(self, x):
#         x = self.flatten(x)
#         logits = self.linear_relu_stack(x)
#         return logits

class LNN1(nn.Module):  # aka the linear model
    def __init__(self, input_size=3 * 32 * 32):
        super(LNN1, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(input_size, 10, bias=False)
        self.bn_1 = nn.BatchNorm1d(input_size, affine=not args.freeze_gamma, track_running_stats=args.ema)
        self.bn_1_no_affine = nn.BatchNorm1d(input_size, affine=False, track_running_stats=args.ema)
        
        if not args.bn:
            self.linear_relu_stack = nn.Sequential(
                self.linear_1
            )
            self.feature_stack = nn.Sequential(
                nn.Identity()
            )
        else:

            self.linear_relu_stack = nn.Sequential(
                self.bn_1,
                self.linear_1
            )
            self.feature_stack = nn.Sequential(
                self.bn_1_no_affine
            )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
    def get_features(self, x):
        x = self.flatten(x)
        return self.feature_stack(x)


class BNLNN1(nn.Module): # aka linear model
    def __init__(self):
        super(BNLNN1, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(3 * 32 * 32, 10, bias=False)
        self.bn_1 = nn.BatchNorm1d(10, track_running_stats=args.ema)
        if not args.bn:
            self.linear_relu_stack = nn.Sequential(
                self.linear_1
            )
        else:
            self.linear_relu_stack = nn.Sequential(
                self.linear_1,
                self.bn_1
            )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# class LNN2(nn.Module):
#     def __init__(self):
#         super(LNN2, self).__init__()
#         self.flatten = nn.Flatten()
#         self.linear_1 = nn.Linear(args.width, 10, bias=False)
#         self.linear_2 = nn.Linear(3 * 32 * 32, args.width, bias=False)

#         self.bn_1 = nn.BatchNorm1d(args.width, track_running_stats=args.ema)
#         if not args.bn:
#             self.linear_relu_stack = nn.Sequential(
#                 self.linear_2,
#                 self.linear_1
#             )
#         else:
#             self.linear_relu_stack = nn.Sequential(
#                 self.linear_2,
#                 self.bn_1, 
#                 self.linear_1
#             )

#     def forward(self, x):
#         x = self.flatten(x)
#         logits = self.linear_relu_stack(x)
#         return logits

class LNN2(nn.Module):
    def __init__(self, input_size=3 * 32 * 32):
        super(LNN2, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(args.width, 10, bias=False)
        self.linear_2 = nn.Linear(input_size, args.width, bias=False)

        self.bn_1 = nn.BatchNorm1d(args.width, affine=not args.freeze_gamma, track_running_stats=args.ema)
        self.bn_1_no_affine = nn.BatchNorm1d(args.width, affine=False, track_running_stats=args.ema)

        if not args.bn:
            self.linear_relu_stack = nn.Sequential(
                self.linear_2,
                self.linear_1
            )
            self.feature_stack = nn.Sequential(
                self.linear_2,
            )
        else:
            self.linear_relu_stack = nn.Sequential(
                self.linear_2,
                self.bn_1, 
                self.linear_1
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
    def __init__(self):
        super(LNN3, self).__init__()
        self.flatten = nn.Flatten()
        
        self.linear_1 = nn.Linear(args.width, 10, bias=False)
        self.linear_2 = nn.Linear(args.width, args.width, bias=False)
        self.linear_3 = nn.Linear(3 * 32 * 32, args.width, bias=False)

        self.bn_1 = nn.BatchNorm1d(args.width, track_running_stats=args.ema)
        self.bn_2 = nn.BatchNorm1d(args.width, track_running_stats=args.ema)
        self.bn_1_no_affine = nn.BatchNorm1d(args.width, affine=False, track_running_stats=args.ema)

        if not args.bn:
            self.linear_relu_stack = nn.Sequential(
                self.linear_3,
                self.linear_2,
                self.linear_1
            )
            self.feature_stack = nn.Sequential(
                self.linear_3,
                self.linear_2,
            )
        else:
            self.linear_relu_stack = nn.Sequential(
                self.linear_3,
                self.bn_2,
                self.linear_2,
                self.bn_1,
                self.linear_1
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


class FC2(nn.Module):
    def __init__(self, input_size = 3 * 32 * 32):
        super(FC2, self).__init__()
        self.flatten = nn.Flatten()

        self.linear_1 = nn.Linear(args.width, 10, bias=True)
        self.linear_2 = nn.Linear(input_size, args.width, bias=True)

        self.bn_1 = nn.BatchNorm1d(args.width, affine=not args.freeze_gamma, track_running_stats=args.ema)
        self.bn_1_no_affine = nn.BatchNorm1d(args.width, affine=False, track_running_stats=args.ema)

        if not args.bn:
            self.linear_relu_stack = nn.Sequential(
                self.linear_2,
                nn.ReLU(),
                self.linear_1,
            )

            self.feature_stack = nn.Sequential(
                self.linear_2,
                nn.ReLU(),
            )
        else:
            self.linear_relu_stack = nn.Sequential(
                self.linear_2,
                self.bn_1,
                nn.ReLU(),
                self.linear_1,
            )

            self.feature_stack = nn.Sequential(
                self.linear_2,
                self.bn_1_no_affine,
                nn.ReLU(),
            )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def get_features(self, x):
        x = self.flatten(x)
        return self.feature_stack(x)

class FC3(nn.Module):
    def __init__(self):
        super(FC3, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(args.width, 10, bias=True)
        self.linear_2 = nn.Linear(args.width, args.width, bias=True)
        self.linear_3 = nn.Linear(3 * 32 * 32, args.width, bias=True)

        self.bn_1 = nn.BatchNorm1d(args.width, track_running_stats=args.ema)
        self.bn_2 = nn.BatchNorm1d(args.width, track_running_stats=args.ema)
        self.bn_1_no_affine = nn.BatchNorm1d(args.width, affine=False, track_running_stats=args.ema)

        if not args.bn:
            self.linear_relu_stack = nn.Sequential(
                self.linear_3,
                nn.ReLU(),
                self.linear_2,
                nn.ReLU(),
                self.linear_1
            )
            self.feature_stack = nn.Sequential(
                self.linear_3,
                nn.ReLU(),
                self.linear_2,
                nn.ReLU(),
            )
        else:
            self.linear_relu_stack = nn.Sequential(
                self.linear_3,
                self.bn_2,
                nn.ReLU(),
                self.linear_2,
                self.bn_1,
                nn.ReLU(),
                self.linear_1
            )
            self.feature_stack = nn.Sequential(
                self.linear_3,
                self.bn_2,
                nn.ReLU(),
                self.linear_2,
                self.bn_1,
                nn.ReLU(),
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
            writer.add_text('seed', str(seed), 1)
            print('seed is', seed)
            self.permutation = torch.randperm(n, generator=generator).tolist()
            yield from self.permutation
        else:
            yield from self.permutation
        self.epoch = self.epoch + 1


def plot_convex_hulls(X, y):
    positive_labels = y == 1
    positive_features = X[positive_labels, :]
    negative_features = X[~positive_labels, :]
    fig, ax = plt.subplots()
    ax.plot(positive_features[:,0], positive_features[:,1], '+', color='b', alpha=0.5, label='Positive')
    ax.plot(negative_features[:,0], negative_features[:,1], 'o', color='r', alpha=0.5, label='Negative')
    positive_hull = ConvexHull(positive_features)
    negative_hull = ConvexHull(negative_features)
    for simplex in positive_hull.simplices:
        ax.plot(positive_features[simplex, 0], positive_features[simplex, 1], 'b--', alpha=0.5)
    for simplex in negative_hull.simplices:
        ax.plot(negative_features[simplex, 0], negative_features[simplex, 1], 'r--', alpha=0.5)
    ax.legend()
    return fig, ax


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 50 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def calcloss(train_eval_dataloader, test_eval_dataloader, model, loss_fn, epochidx, savefilename, save=True):
    """
    Compute FB train/val loss and accuracy
    """
    model.eval()
    size = len(train_eval_dataloader.dataset)
    train_loss, train_correct = 0, 0
    with torch.no_grad():
        for X, y in train_eval_dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            train_loss += loss.item() * len(y)
            train_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    train_loss /= size
    train_correct /= size
    print(f"Training Error: \n Accuracy: {(100 * train_correct):>0.1f}%, Avg loss: {train_loss:>8f} \n")
    if args.eval_type == 'SS':
        writer.add_scalar('SS risk/train', train_loss, epochidx)
        writer.add_scalar('SS acc/train', train_correct, epochidx)
    elif args.eval_type == 'GD':
        writer.add_scalar('GD risk/train', train_loss, epochidx)
        writer.add_scalar('GD acc/train', train_correct, epochidx)


    size = len(test_eval_dataloader.dataset)
    test_loss, test_correct = 0, 0
    with torch.no_grad():
        for X, y in test_eval_dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            test_loss += loss.item() * test_eval_batch_size
            test_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            # visualize data in 2d
            if epochidx % 200 == 0 and args.model != 'resnet_18':
                learned_features = model.get_features(X)
                d = learned_features.shape[-1]
                if d == 2:
                    fig, ax = plot_convex_hulls(learned_features.cpu().numpy(), y.cpu().numpy())
                    ax.set_title(f'2d visualization of features at epoch {epochidx}')
                    writer.add_figure('2d visualization', fig, epochidx)
                # if d <= 3:
                #     pos = learned_features[y==1]
                #     neg = learned_features[y==0]
                #     pos_hull = polytope.qhull(pos)
                #     neg_hull = polytope.qhull(neg)
                #     hull_all = polytope.qhull(np.vstack((pos, neg)))
                #     intersect = polytope.intersect(pos_hull, neg_hull)
                #     union = polytope.union(pos_hull, neg_hull)
                #     iou = intersect.volume/union.volume if union.volume > 0 else 0
                #     nls_metric = (pos_hull.volume + neg_hull.volume) / hull_all.volume if hull_all.volume > 0 else 0
                #     # print(intersect.volume, union.volume)
                #     writer.add_scalar('IOU normalized', np.power(iou, 1/d), epochidx)
                #     writer.add_scalar('IOU unnormalized', iou, epochidx)
                #     writer.add_scalar('NLS metric', nls_metric, epochidx)
                # if d > 2:
                #     M = model.linear_1.weight.numpy() if args.freeze_gamma else model.linear_1.weight.numpy() * model.bn_1.weight.numpy()
                #     Mhat = M.reshape((d, 1)) / np.linalg.norm(M)
                #     orthog_projected_data = learned_features @ (np.identity(d) - Mhat @ Mhat.T)
                #     pca = PCA(n_components=1)
                #     orthog_feature = pca.fit_transform(orthog_projected_data)
                #     magnitude_along_M = learned_features @ Mhat 
                #     transformed_data = np.hstack((magnitude_along_M, orthog_feature))
                #     fig, ax = plot_convex_hulls(transformed_data, y)
                #     ax.set_title(f'Orthogonal decomposition along M with PCA at epoch {epochidx}')
                #     ax.set_xlabel('Magnitude in M direction')
                #     ax.set_ylabel('Magnitude in PCA M^perp direction')
                #     writer.add_figure('Orthog decomposition', fig, epochidx)
                # else:
                #     fig, ax = plot_convex_hulls(learned_features, y)
                #     ax.set_title(f'2d visualization of features at epoch {epochidx}')
                #     writer.add_figure('2d visualization', fig, epochidx)
    test_loss /= size
    test_correct /= size
    print(f"Test Error: \n Accuracy: {(100 * test_correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    if args.eval_type == 'SS':
        writer.add_scalar('GD risk/train', test_loss, epochidx)
        writer.add_scalar('GD acc/train', test_correct, epochidx)
    elif args.eval_type == 'GD':
        writer.add_scalar('GD risk/test', test_loss, epochidx)
        writer.add_scalar('GD acc/test', test_correct, epochidx)

    Gamma = None
    W = None
    
    if args.model != 'resnet_18':
        # log weight norms every 100 epochs
        if epochidx % 100 == 0:
            for name, param in model.named_parameters():
                param_size = np.sqrt(torch.numel(param)) # sqrt is the correct normalization here
                writer.add_scalar(f'{name} norm normalized by size', param.norm() / param_size, epochidx)
                writer.add_scalar(f'{name} norm unnormalized', param.norm(), epochidx)
                if name == 'bn_1.weight':
                    Gamma = param
                    # writer.add_scalars('Gamma weights', {str(i): Gamma[i].item() for i in range(Gamma.shape[0])}, epochidx)
                if name == 'linear_1.weight':
                    W = param
                    # writer.add_scalars('W weights', {f'{i}, {j}': W[i][j].item() for i in range(W.shape[0]) for j in range(W.shape[1])}, epochidx)

            M = W @ Gamma
            # writer.add_scalars('M weights', {f'{i}, {j}': M[i][j].item() for i in range(M.shape[0]) for j in range(M.shape[1])}, epochidx)
            writer.add_scalar(f'M norm normalized by size', M.norm() / np.sqrt(torch.numel(M)), epochidx)
            writer.add_scalar(f'M norm unnormalized', M.norm(), epochidx)
    else:
        M = model.fc.weight
        writer.add_scalar(f'M norm normalized by size', M.norm() / np.sqrt(torch.numel(M)), epochidx)
        writer.add_scalar(f'M norm unnormalized', M.norm(), epochidx)
    if save:
        f = open(savefilename, "a")
        f.write(
            str(epochidx) + " " + str(train_loss) + " " + str(train_correct) + " " + str(test_loss) + " " + str(
                test_correct))
        f.write("\n")
        f.close()



if __name__ == '__main__':
    start_time = time.time()
    os.makedirs('./results/', exist_ok=True)
    os.makedirs('./model/', exist_ok=True)

    if args.resume <= 0:
        filename = 'cifar10'
        for key, value in vars(args).items():
            if key == 'gpu' or key == 'resume':
                continue
            else:
                filename = filename + '_' + str(value)
        print('Results will be stored in ./results/' + filename + '.txt')
    else:
        oldfilename = 'cifar10'
        filename = 'cifar10'
        for key, value in vars(args).items():
            if key == 'gpu' or key == 'resume':
                continue
            elif key == 'epoch':
                oldfilename = oldfilename + '_' + str(args.resume)
                filename = filename + '_' + str(args.epoch + args.resume)
            else:
                oldfilename = oldfilename + '_' + str(value)
                filename = filename + '_' + str(value)
        print('Loading checkpoint stored in ./model/' + oldfilename + '.pth and resume')
        print('Results will be stored in ./results/' + filename + '.txt')

        checkpoint = torch.load('./model/' + oldfilename + '.pth')
        fold = open('./results/' + oldfilename + '.txt', 'r')
        fnew = open('./results/' + filename + '.txt', 'a')
        lines = fold.readlines()
        for i in range(len(lines) - 1):
            fnew.write(lines[i])
        fold.close()
        fnew.close()

    # Get cpu or gpu device for training.
    device = 'cuda:' + str(args.gpu) if torch.cuda.is_available() else "cpu"
    # device = 'cpu'
    print('Using {} device'.format(device))
    # Define model.
    if args.model == '1l_lnn':
        model = LNN1().to(device)
    elif args.model == 'bn_1l_lnn':
        model = BNLNN1().to(device)
    elif args.model == '2l_lnn':
        model = LNN2().to(device)
    elif args.model == '3l_lnn':
        model = LNN3().to(device)
    elif args.model == '2l_fc':
        model = FC2().to(device)
    elif args.model == '3l_fc':
        model = FC3().to(device)
    elif args.model == 'resnet_18':
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        model.fc = nn.Linear(512, args.num_classes)
        model = model.to(device)
    elif args.model == 'vgg_16':
        weights = VGG16_BN_Weights.DEFAULT
        model = vgg16_bn(weights=weights)
        for param in model.features.parameters():
            param.requires_grad = False

        num_features = model.classifier[6].in_features
        features = list(model.classifier.children())[:-1] # Remove last layer
        features.extend([nn.Linear(num_features, args.num_classes)]) # Add our layer with 4 outputs
        model.classifier = nn.Sequential(*features)
        model = model.to(device)
    else:
        raise ValueError('--model argument should be one of 1l_lnn, 2l_lnn, 3l_lnn, 2l_fc, 3l_fc.')
    print(model)

    if args.model == 'resnet_18':
        transform = weights.transforms()
    elif args.dn:
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        transform = transforms.ToTensor()

    # Download training data from open datasets.
    training_data = torchvision.datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=transform,
    )

    # Download test data from open datasets.
    test_data = torchvision.datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=transform,
    )

    batch_size = args.bsize
    train_eval_batch_size = args.bsize if args.model == 'resnet_18' else len(training_data)
    test_eval_batch_size = args.bsize if args.model == 'resnet_18' else len(test_data)

    # Create data loaders.
    custom_sampler = None
    sampling = args.sampling
    if sampling == 'SGD':
        custom_sampler = RandomSampler(data_source=training_data, replacement=True)
    elif sampling == 'SS':
        custom_sampler = RandomSamplerSS(training_data)
    elif sampling == 'RR':
        custom_sampler = RandomSampler(data_source=training_data)
    else:
        raise ValueError('--sampling argument should be one of SGD, SS or RR.')

    train_dataloader = DataLoader(training_data, batch_size=batch_size, sampler=custom_sampler, num_workers=2)
    train_plot_dataloader = DataLoader(training_data, batch_size=train_eval_batch_size, num_workers=1)
    test_plot_dataloader = DataLoader(test_data, batch_size=test_eval_batch_size, num_workers=1)

    

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momen)

    if args.resume <= 0:
        if args.eval_type == 'GD':
            calcloss(train_plot_dataloader, test_plot_dataloader, model, loss_fn, 0, "./results/" + filename + ".txt")
        elif args.eval_type == 'SS':
            calcloss(train_dataloader, train_plot_dataloader, model, loss_fn, 0, "./results/" + filename + ".txt")


        for t in range(args.epoch):
            print(f"Epoch {t + 1}\n-------------------------------")
            train(train_dataloader, model, loss_fn, optimizer)
            if args.eval_type == 'GD':
                calcloss(train_plot_dataloader, test_plot_dataloader, model, loss_fn, t+1, "./results/" + filename + ".txt")
            elif args.eval_type == 'SS':
                calcloss(train_dataloader, test_plot_dataloader, model, loss_fn, t+1, "./results/" + filename + ".txt")
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if args.sampling == 'SS':
            custom_sampler.permutation = checkpoint['SS_perm']
            custom_sampler.epoch = args.resume

        calcloss(train_plot_dataloader, test_plot_dataloader, model, loss_fn, args.resume,
                 "./results/" + filename + ".txt", False)
        for t in range(args.resume, args.resume + args.epoch):
            print(f"Epoch {t + 1}\n-------------------------------")
            train(train_dataloader, model, loss_fn, optimizer)
            calcloss(train_plot_dataloader, test_plot_dataloader, model, loss_fn, t + 1,
                     "./results/" + filename + ".txt")

    print("Done!")

    sfn = "./model/" + filename + ".pth"

    if sampling == 'SS':
        state = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'SS_perm': custom_sampler.permutation
        }
    else:
        state = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
    torch.save(state, sfn)
    print("Saved PyTorch Model State to " + sfn)
    writer.flush()
    writer.close()
    f = open("./results/" + filename + ".txt", "a")
    f.write("Execution time: %s seconds" % (time.time() - start_time))
    f.close()
