#%%
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat
from torchvision import models
import torchvision.transforms as transforms
import pandas as pd
from utils_local import get_norm_layer, get_non_linearity

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#-----------------------------------------------------------------------
#   Define the residual network 
#-----------------------------------------------------------------------
class ResidualPhi(nn.Module):
    def __init__(self, nblocks, dim=256, nl_layer='relu', norm_layer='batch1d', n_branches=1):
        super(ResidualPhi, self).__init__()
        self.blocks = nn.ModuleList([ResBlock(dim, nl_layer, norm_layer, n_branches) for _ in range(nblocks)])

    def forward(self, x):
        rs = []
        for block in self.blocks:
            x, r = block(x)
            rs.append(r)
        return x, rs

    def backward(self, y, maxIter=10):
        x = y
        for block in self.blocks:
            x = block.backward(x, maxIter=maxIter)
        return x


class ResBlock(nn.Module):
    def __init__(self, dim, nl_layer='relu', norm_layer='batch1d', n_branches=1):
        super(ResBlock, self).__init__()
        self.coeff = 0.9
        self.n_power_iter = 1

        branches = []
        branches += [nn.Linear(int(dim), int(dim))]

        for i in range(n_branches):
            if norm_layer != 'none':
                branches += [get_norm_layer(norm_layer)(int(dim))]
            if nl_layer != 'none':
                branches += [get_non_linearity(nl_layer)()]
            branches += [nn.Linear(int(dim), int(dim))]

        self.branches = nn.Sequential(*branches)

    def forward(self, x):
        r = self.branches(x)
        return x + r, r

    def backward(self, y, maxIter=10):
        x = y
        for iter_index in range(maxIter):
            summand = self.branches(x)
            x = y - summand
        return x

#-----------------------------------------------------------------------
#  
#-----------------------------------------------------------------------


class DomainClassifier(nn.Module):
    def __init__(self, input_dim=256,n_hidden=100):
        super(DomainClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, n_hidden, bias=True)
        self.fc2 = nn.Linear(n_hidden, n_hidden, bias=True)
        self.fc3 = nn.Linear(n_hidden, 1, bias=True)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, input):
        x = self.activation(self.fc1(input))
        x = self.activation(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x



class FeatureExtractor(nn.Module):
    def __init__(self, input_dim=100,n_hidden=256,output_dim=256):
        super(FeatureExtractor, self).__init__()
        self.fc1 = nn.Linear(input_dim, n_hidden)
        self.fc2 = nn.Linear(n_hidden, output_dim)
        self.activation = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, input):
        x = self.activation(self.fc1(input))
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        return x


class DataClassifier(nn.Module):
    def __init__(self, input_dim=256,n_class=10):
        super(DataClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim,n_class)

    def forward(self, input):
        x = (self.fc1(input.view(input.size(0), -1))) 
        return x



# ------------------------------------------------------------------------------
#                   Digits
# ------------------------------------------------------------------------------


class FeatureExtractorDigits(nn.Module):
    def __init__(self, channel, kernel_size=5, output_dim=128):
        super(FeatureExtractorDigits, self).__init__()
        self.conv1 = nn.Conv2d(channel, 64, kernel_size=kernel_size)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=kernel_size)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, output_dim, kernel_size=kernel_size)
        self.bn3 = nn.BatchNorm2d(output_dim)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, input):
        x = self.bn1(self.conv1(input))
        x = self.act(self.pool1(x))
        x = self.bn2(self.conv2(x))
        x = self.act(self.pool2(x))
        x = self.bn3(self.conv3(x))
        x = x.view(x.size(0), -1)
        return x


class DataClassifierDigits(nn.Module):
    def __init__(self, n_class, input_size=128):
        super(DataClassifierDigits, self).__init__()
        self.fc1 = nn.Linear(input_size, 100)
        self.bn1 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, 100)
        self.bn2 = nn.BatchNorm1d(100)
        self.fc3 = nn.Linear(100, n_class)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, input):
        x = self.act(self.bn1(self.fc1(input)))
        x = self.act(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    feat = FeatureExtractorDigits(1, 3, 128)
    
    print(feat(torch.randn(10, 1, 28, 28)).shape)


# %%
