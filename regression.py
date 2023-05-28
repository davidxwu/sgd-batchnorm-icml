from argparse import Namespace
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import RandomSampler, SequentialSampler
import time
import torchvision
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn import svm
from sklearn.datasets import make_blobs, make_moons, make_regression
from sklearn.model_selection import train_test_split
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from matplotlib import cm
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch Regression Training')
parser.add_argument('--model', default='2l_lnn', type=str, help='model, one of bn_1l_lnn, 1l_lnn, 2l_lnn, 3l_lnn, 2l_fc, 3l_fc')
parser.add_argument('--fixed_w', default=False, action='store_true', help='use fixed relationship')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--bsize', default=128, type=int, help='minibatch size')
parser.add_argument('--num_datapoints', default=10000, type=int, help='size of dataset')
parser.add_argument('--num_dimensions', default=10, type=int, help='input dimensionality')
parser.add_argument('--sampling', default="RR", type=str, help='sampling type, one of RR, SS, IGM, or SGD')
parser.add_argument('--seed', default=-1, type=int, help='seed for randomness (nondet if not included)')
parser.add_argument('--width', default=128, type=int, help='hidden layer width')
parser.add_argument('--epoch', default=100, type=int, help='number of training epochs')
parser.add_argument('--bn', dest='bn', default=False, action='store_true', help='use batchnorm layers')
parser.add_argument('--freeze_gamma', default=False, action='store_true', help='use bn but freeze gamma')
parser.add_argument('--dn', dest='dn', default=False, action='store_true', help='data normalization. [0,1] to [-1,1]')
parser.add_argument('--momen', default=0.0, type=float, help='momentum param of SGD')
parser.add_argument('--rn', default=1, type=int, help='run number')  # just to keep log files separate
parser.add_argument('--gpu', default=0, type=int, help='designate GPU number')
parser.add_argument('--resume', default=0, type=int, help='read and resume')
parser.add_argument('--suffix', default='', type=str, help='suffix for log file')
args: Namespace = parser.parse_args()


writer = SummaryWriter(comment=f'_regression_shuffling_{args.sampling}_lr_{args.lr}_epoch_{args.epoch}_model_{args.model}_num_datapoints_{args.num_datapoints}_bsize_{args.bsize}_bn_{args.bn}_freeze_{args.freeze_gamma}_momen_{args.momen}_num_dimensions_{args.num_dimensions}_seed_{args.seed}_fixed_w_{args.fixed_w}_{args.suffix}')

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

# Define models
class LNN1(nn.Module):  # aka the linear model
    def __init__(self, input_size=2, output_dim=1):
        super(LNN1, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(input_size, output_dim, bias=False)
        self.bn_1 = nn.BatchNorm1d(input_size, affine=not args.freeze_gamma, track_running_stats=False)
        self.bn_1_no_affine = nn.BatchNorm1d(input_size, affine=False, track_running_stats=False)
<<<<<<< HEAD
        with torch.no_grad():
            nn.init.zeros_(self.linear_1.weight)
            # nn.init.zeros_(self.linear_1.bias)
=======
        nn.init.zeros_(self.linear_1.weight)
>>>>>>> 3a3948484c5fb4145d17c27b0af663d28b087315

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
        # # print(x.shape)
        # bn_x = self.bn_1(x)
        # print('bn', bn_x)
        # logits = self.linear_1(x)
        # print('M', self.linear_1.weight, 'output', logits)
        logits = self.linear_relu_stack(x)
        return logits
    
    def get_features(self, x):
        x = self.flatten(x)
        return self.feature_stack(x)


class BNLNN1(nn.Module): # aka BN(WX)
    def __init__(self, input_size=2):
        super(BNLNN1, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(input_size, 1, bias=False)
        self.bn_1 = nn.BatchNorm1d(1, track_running_stats=False)
        self.bn_1_no_affine = nn.BatchNorm1d(1, affine=False, track_running_stats=False)

        if not args.bn:
            self.linear_relu_stack = nn.Sequential(
                self.linear_1
            )
            self.feature_stack = nn.Sequential(
                nn.Identity()
            )
        else:
            self.linear_relu_stack = nn.Sequential(
                self.linear_1,
                self.bn_1
            )
            self.feature_stack = nn.Sequential(
                self.linear_1,
                self.bn_1_no_affine
            )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def get_features(self, x):
        x = self.flatten(x)
        return self.feature_stack(x)

class LNN2(nn.Module):
    def __init__(self, input_size=2, output_dim=1):
        super(LNN2, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(args.width, output_dim, bias=False)
        self.linear_2 = nn.Linear(input_size, args.width, bias=False)

        self.bn_1 = nn.BatchNorm1d(args.width, affine=not args.freeze_gamma, track_running_stats=False)
        self.bn_1_no_affine = nn.BatchNorm1d(args.width, affine=False, track_running_stats=False)

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
    def __init__(self, input_size=2, output_dim=1):
        super(LNN3, self).__init__()
        self.flatten = nn.Flatten()
        
        self.linear_1 = nn.Linear(args.width, output_dim, bias=False)
        self.linear_2 = nn.Linear(args.width, args.width, bias=False)
        self.linear_3 = nn.Linear(input_size, args.width, bias=False)

        self.bn_1 = nn.BatchNorm1d(args.width, track_running_stats=False)
        self.bn_1_no_affine = nn.BatchNorm1d(args.width, affine=False, track_running_stats=False)
        self.bn_2 = nn.BatchNorm1d(args.width, track_running_stats=False)
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


class RandomSamplerSS(RandomSampler):
    def __init__(self, train_dataset, permutation=None):
        super().__init__(train_dataset)
        self.epoch = 1
        self.permutation = permutation

    def __iter__(self):
        n = len(self.data_source)
        if self.epoch == 1 and self.permutation is None:
            generator = torch.Generator()
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator.manual_seed(seed)
            writer.add_text('seed', str(seed), 1)
            self.permutation = torch.randperm(n, generator=generator).tolist()
            yield from self.permutation
        else:
            yield from self.permutation
        # print(self.permutation)
        self.epoch = self.epoch + 1

class RegressionDataset(Dataset):
    def __init__(self, num_samples=100, num_dimensions=1, seed=None, transform=None, target_transform=None):
        # self.X, self.y, self.w = make_regression(n_samples=num_samples, n_features=num_dimensions, n_informative=num_dimensions, noise=1, coef=True)
        if seed is None or seed == -1:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
        
        if args.fixed_w:
            rg = np.random.Generator(np.random.PCG64(seed))
            self.X = rg.standard_normal((num_samples, num_dimensions))
            self.w = np.ones(num_dimensions)
            self.y = self.X @ self.w
        else:
            self.X, self.y, self.w = make_regression(n_samples=num_samples, n_features=num_dimensions, n_informative=num_dimensions, noise=1, coef=True, random_state=seed)
        self.X = torch.Tensor(self.X).float()
        self.y = torch.Tensor(self.y).float()
        self.transform = transform
        self.target_transform = target_transform
        if self.transform is not None:
            self.X = self.transform(self.X).float()
        if self.target_transform is not None:
            self.y = self.target_transform(self.y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx, :], self.y[idx]
<<<<<<< HEAD
def train(dataloader, model, loss_fn, optimizer, scheduler):
=======
def train(dataloader, model, loss_fn, optimizer):
>>>>>>> 3a3948484c5fb4145d17c27b0af663d28b087315
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred.squeeze(), y)

        # print('train', model.get_features(X), pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 50 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    scheduler.step()

def calcloss(fb_train_dataloader, sampling_train_dataloader, model, loss_fn, epochidx, savefilename, save=True):
    model.eval()
    fb_size = len(fb_train_dataloader.dataset)
    fb_loss = 0
    distorted_loss = 0
    X_fb = None
    y_fb = None
    list_ss_features = []
    list_y = []
    list_pred = []
    with torch.no_grad():
        for X, y in fb_train_dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            X_fb = model.get_features(X)
            y_fb = y
            # loss = loss_fn(pred.squeeze(), y)
            # # print('fb test', X, pred, y)
            # fb_loss += loss.item() * fb_size
            # plt.scatter(features, y, color='b', label='True')
            # plt.scatter(features, pred, color='r', marker='^', label='Predicted')
            # plt.title('FB data')
            # plt.legend()
            # plt.show()

        for i, (X, y) in enumerate(sampling_train_dataloader):
            X, y = X.to(device), y.to(device)
            ss_pred = model(X)
            ss_features = model.get_features(X)
            list_ss_features.append(ss_features.cpu().numpy())
            # print(ss_features @ m)
            list_y.append(y.cpu().numpy().reshape(-1, 1))
            ss_loss = loss_fn(ss_pred.squeeze(), y.squeeze())
            list_pred.append(ss_pred.cpu().numpy())
            # print('ss pred actual', ss_pred)
            M = (model.linear_1.weight * model.bn_1.weight).cpu().numpy().reshape(-1, 1) if not args.freeze_gamma else (model.linear_1.weight).cpu().numpy().squeeze()
            # print(ss_features.cpu().numpy() @ M)
            # print('distorted', ss_features, pred, y)
            # plt.scatter(ss_features, y,label=f'True batch {i}', alpha=0.5)
            # plt.scatter(ss_features, pred, marker='^', label=f'Predicted batch {i}', alpha=0.5)
            # distorted_loss += ss_loss.item() * len(y)
        
        # plt.title('SS data')
        # plt.legend()
        # plt.show()
            
        # if epochidx % 1000 == 0:
        #     

    # print(list_pred)
    X_pi = np.vstack(list_ss_features)
    y_pi = np.vstack(list_y)
    pred_pi = np.vstack(list_pred)
    fb_loss = np.linalg.norm(X_fb @ M - y_fb) ** 2 / fb_size
    distorted_loss = np.linalg.norm(X_pi @ M - y_pi) ** 2 / fb_size
    print(f"Training Error: \n FB loss: {fb_loss:>8f}, Distorted loss: {distorted_loss:>8f} \n")
    
    writer.add_scalar('fb loss/train', fb_loss, epochidx)
    writer.add_scalar('distorted loss/train', distorted_loss, epochidx)
    # writer.add_scalar('fb acc/train', train_correct, epochidx)
    
    

    # print(X_pi)

    with torch.no_grad():
        # log weight norms every 100 epochs
        
        M = (model.linear_1.weight * model.bn_1.weight).cpu().numpy().reshape(-1) if not args.freeze_gamma else (model.linear_1.weight).cpu().numpy().squeeze()
        # print('ok now', M)
        M_fb = np.linalg.lstsq(X_fb.cpu().numpy(), y_fb.cpu().numpy())[0]
        # print(M_fb.shape)
        writer.add_scalar(f'normalized distance to GD optimum', np.linalg.norm(M - M_fb)/np.linalg.norm(M_fb), epochidx)
        M_pi, ell_pi_star = np.linalg.lstsq(X_pi, y_pi.squeeze())[:2]
        # print(f'Analytical {M_pi} actual {M}')
        # print(X_pi @ M_pi)
        # print(X_pi @ M)
        # print(pred_pi)
        writer.add_scalar(f'opt SS loss', ell_pi_star / len(y_pi), epochidx)
        writer.add_scalar(f'normalized distance to SS optimum', np.linalg.norm(M- M_pi) / np.linalg.norm(M_pi), epochidx)
        # print(X_pi.shape, M.shape, M_pi.shape)
        # print(np.linalg.norm(X_pi @ M - y_pi)**2/fb_size, np.linalg.norm(X_pi @ M_pi.reshape(-1, 1) - y_pi)**2/fb_size)
    if save:
        f = open(savefilename, "a")
        f.write(
            str(epochidx) + " " + str(fb_loss) + " " + str(distorted_loss))
        f.write("\n")
        f.close()
    
    return ell_pi_star / len(y_pi)



if __name__ == '__main__':
    start_time = time.time()
    os.makedirs('./results/', exist_ok=True)
    os.makedirs('./model/', exist_ok=True)

    if args.resume <= 0:
        filename = 'toy_data'
        for key, value in vars(args).items():
            if key == 'gpu' or key == 'resume':
                continue
            else:
                filename = filename + '_' + str(value)
        print('Results will be stored in ./results/' + filename + '.txt')
    else:
        oldfilename = 'toy_data'
        filename = 'toy_data'
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

    if args.dn:
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        transform = transforms.ToTensor()

    
    input_size = args.num_dimensions
    training_data = RegressionDataset(num_samples=args.num_datapoints, num_dimensions=input_size, seed=args.seed)
    # test_data = RegressionDataset(num_samples=args.num_datapoints, num_dimensions=input_size)
    

    batch_size = args.bsize
    ss_eval_batch_size = batch_size
    train_eval_batch_size = len(training_data)
    # test_eval_batch_size = len(test_data)

    # Create data loaders.
    custom_sampler = None
    sampling = args.sampling
    if sampling == 'SGD':
        custom_sampler = RandomSampler(data_source=training_data, replacement=True)
    elif sampling == 'SS':
        custom_sampler = RandomSamplerSS(training_data)
    elif sampling == 'RR':
        custom_sampler = RandomSampler(data_source=training_data)
    elif sampling == 'IGM':
        custom_sampler = SequentialSampler(data_source=training_data)
    else:
        raise ValueError('--sampling argument should be one of SGD, SS or RR.')
    

    train_dataloader = DataLoader(training_data, batch_size=batch_size, sampler=custom_sampler, num_workers=2)
    # ss_train_dataloader = DataLoader(training_data, batch_size=ss_eval_batch_size, sampler=custom_sampler, num_workers=2)
    fb_train_dataloader = DataLoader(training_data, batch_size=train_eval_batch_size, num_workers=2)
    # test_plot_dataloader = DataLoader(test_data, batch_size=test_eval_batch_size, num_workers=2)

    # Get cpu or gpu device for training.
    torch.cuda.is_available = lambda : False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cuda:' + str(args.gpu) if torch.cuda.is_available() else "cpu"
    print('Using {} device'.format(device))

    # Define model.
    if args.model == '1l_lnn':
        model = LNN1(input_size=input_size).to(device)
    elif args.model == 'bn_1l_lnn':
        model = BNLNN1(input_size=input_size).to(device)
    elif args.model == '2l_lnn':
        model = LNN2(input_size=input_size).to(device)
    elif args.model == '3l_lnn':
        model = LNN3(input_size=input_size).to(device)
    elif args.model == '2l_fc':
        model = FC2().to(device)
    elif args.model == '3l_fc':
        model = FC3().to(device)
    else:
        raise ValueError('--model argument should be one of 1l_lnn, 2l_lnn, 3l_lnn, 2l_fc, 3l_fc.')
    print(model)

    print(model.linear_1.bias)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momen)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    if args.resume <= 0:
        calcloss(fb_train_dataloader, train_dataloader, model, loss_fn, 0, "./results/" + filename + ".txt")

        for t in range(args.epoch):
            print(f"Epoch {t + 1}\n-------------------------------")
            train(train_dataloader, model, loss_fn, optimizer, scheduler)
            mse_pi_star = calcloss(fb_train_dataloader, train_dataloader, model, loss_fn, t + 1,
                     "./results/" + filename + ".txt")
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if args.sampling == 'SS':
            custom_sampler.permutation = checkpoint['SS_perm']
            custom_sampler.epoch = args.resume

        calcloss(fb_train_dataloader, train_dataloader, model, loss_fn, args.resume,
                 "./results/" + filename + ".txt", False)
        for t in range(args.resume, args.resume + args.epoch):
            print(f"Epoch {t + 1}\n-------------------------------")
            train(train_dataloader, model, loss_fn, optimizer, scheduler)
            calcloss(fb_train_dataloader, train_dataloader, model, loss_fn, t + 1,
                     "./results/" + filename + ".txt")

    print("Done!")
    print('Optimal SS loss', mse_pi_star)
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
    writer.add_hparams(vars(args), {})
    writer.flush()
    writer.close()
    f = open("./results/" + filename + ".txt", "a")
    f.write("Execution time: %s seconds" % (time.time() - start_time))
    f.close()

