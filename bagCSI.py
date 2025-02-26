
#%%
import math
import os
import time
from scipy.io import arff
import numpy as np
import ot 
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
from data import get_toy
from utils import extract_feature
from models import FullyConnectedNN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def bagCSI_train(model, source_loader, target_bags, n_classes, num_epochs=100,device='cpu',
                    param_bag=0.1, param_da=0.1,
                    learning_rate=0.001,
                    verbose=False):
    model.train()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        loss_epoch = 0
        bag_loss_epoch = 0
        loss_source_epoch = 0
        for i, (x_train, y_train) in enumerate(source_loader):
            x_train = x_train.to(device).float()
            y_train = y_train.to(device)
            i_bag = np.random.randint(0, len(target_bags))
            
            x_target = target_bags[i_bag]['data'].to(device).float()
            y_target_prop = torch.tensor(target_bags[i_bag]['prop']).to(device)

            # source loss
            outputs, source_feature = model(x_train, get_feature=True)
            loss_source = criterion(outputs, y_train)

            # bag loss
            outputs_target,target_feature  = model(x_target, get_feature=True)
            outputs_target = torch.softmax(outputs_target, dim=1)
            loss_bag = torch.mean(torch.abs(outputs_target.mean(dim=0) - y_target_prop))

            # domain adaptation loss as in Equation (12)
            # written for multidimensional regression based on one-hot encoding 
            loss_da = 0
            
            for j in range(n_classes):
                source_feature_data = source_feature*(y_train==j).float().view(-1,1)
                target_feature_data = target_feature*(y_target_prop[j]).float().view(-1,1)
                loss_da += torch.mean(torch.abs(source_feature_data.mean(dim=0) - target_feature_data.mean(dim=0)))**2
                



            optimizer.zero_grad()
            loss = loss_source + param_bag*loss_bag + param_da*loss_da
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()
            loss_source_epoch += loss_source.item()
            bag_loss_epoch += loss_bag.item()           
        
        loss_epoch /= len(source_loader)
        bag_loss_epoch /= len(source_loader)
        loss_source_epoch /= len(source_loader)
        if verbose:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss_epoch:.4f} loss_source: {loss_source_epoch:.4f} loss_bag: {bag_loss_epoch:.4f}')





if __name__ == '__main__':

    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

    dim = 2

    source_loader, target_bags = get_toy(apply_miss_feature_source=True,
                                        dim=dim,
                                        data_variance=0.5,
                                        center_translation=10)

    from data import extract_data_label
    x_test, y_test = extract_data_label(target_bags)
    x_test = x_test.to(device).float()

    # plot source data
    plt.figure(figsize=(5, 4))
    plt.scatter(source_loader.dataset.tensors[0][:, 0], source_loader.dataset.tensors[0][:, 1],
                c=source_loader.dataset.tensors[1], cmap='viridis')

    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap='viridis', marker='x')

    #%%

    input_size = dim
    hidden_size1 = 128
    hidden_size2 = 64
    output_size = 3
    learning_rate = 0.001
    num_epochs = 200

    # Initialize the model, loss function, and optimizer
    model = FullyConnectedNN(input_size, hidden_size1, hidden_size2, output_size)
    bagCSI_train(model, source_loader, target_bags, n_classes=3, num_epochs=num_epochs,device=device,
                    param_bag=1, param_da=1)


    
    from utils import evaluate_clf, create_data_loader
    test_loader = create_data_loader(x_test, y_test, batch_size=128, shuffle=False,drop_last=False)
    acc, bal_acc, cm = evaluate_clf(model, test_loader,n_classes=3,return_pred=False)



    #%%

    train_feat, train_label = extract_feature(model, source_loader, device=device)

    from torchdr import PCA, TSNE
    N=1000
    list_data = [train_feat, xt_feat[:N]]
    stacked_data = torch.cat(list_data,0)
    z = TSNE(perplexity=30,n_components=2).fit_transform(stacked_data)

    x_train_proj = z[:train_feat.size(0)]
    x_t_proj = z[train_feat.size(0):]

    plt.figure(figsize=(5, 4))
    plt.scatter(x_train_proj[:, 0], x_train_proj[:, 1], c=train_label, cmap='viridis')
    plt.scatter(x_t_proj[:, 0], x_t_proj[:, 1], c=y_test[:N], cmap='viridis', marker='x')





# %%
