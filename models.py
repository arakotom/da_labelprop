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
        self.activation = nn.GELU()

    def forward(self, input):
        x = self.activation(self.fc1(input))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
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



# Define the neural network class
class FullyConnectedNN(nn.Module):
    def __init__(self, input_size, n_hidden=256, n_class=10):
        super(FullyConnectedNN, self).__init__()
        # Define layers
        self.fc1 = nn.Linear(input_size, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_class)
        self.relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x, get_feature=False):
        # Define forward pass
        x = self.relu(self.fc1(x))
        x_feat = self.relu(self.fc2(x))
        x_feat = self.dropout(x_feat)
        x = self.fc3(x_feat)  # No softmax here
        if get_feature:
            return x, x_feat
        else:
            return x