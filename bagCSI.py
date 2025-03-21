
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def bagCSI_train(feature_extractor,classifier, source_loader, target_bags, n_classes, num_epochs=100,device='cpu',
                    param_bag=0.1, param_da=0.1,
                    learning_rate=0.001,
                    verbose=False):
    feature_extractor.train()
    feature_extractor.to(device)
    classifier.train()
    classifier.to(device)
    criterion = nn.CrossEntropyLoss()
    optim_feat = optim.Adam(feature_extractor.parameters(), lr=learning_rate,betas=(0.9, 0.999))
    optim_clf = optim.Adam(classifier.parameters(), lr=learning_rate,betas=(0.9, 0.999))
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
            #outputs, source_feature = model(x_train, get_feature=True)
            #loss_source = criterion(outputs, y_train)
            source_feature = feature_extractor(x_train)
            outputs = classifier(source_feature)
            loss_source = criterion(outputs, y_train)

            # bag loss
            #outputs_target,target_feature  = model(x_target, get_feature=True)
            target_feature = feature_extractor(x_target)
            outputs_target = classifier(target_feature)
            outputs_target = torch.softmax(outputs_target, dim=1)
            loss_bag = torch.mean(torch.abs(outputs_target.mean(dim=0) - y_target_prop))

            # domain adaptation loss as in Equation (12)
            # written for multidimensional regression based on one-hot encoding 
            loss_da = 0
            for j in range(n_classes):
                source_feature_data = source_feature*(y_train==j).float().view(-1,1)
                target_feature_data = target_feature*(y_target_prop[j]).float().view(-1,1)
                loss_da += torch.mean(torch.abs(source_feature_data.mean(dim=0) - target_feature_data.mean(dim=0)))**2
                



            optim_feat.zero_grad()
            optim_clf.zero_grad()
            loss = loss_source + param_bag*loss_bag + param_da*loss_da
            loss.backward()
            optim_feat.step()
            optim_clf.step()
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
    
    if 0:

        dim = 2
        n_class = 3
        n_hidden = 128
        num_epochs  = 300
        lr = 0.001

        source_loader, target_bags = get_toy(apply_miss_feature_source=True,
                                            dim=dim,
                                            data_variance=0.5,
                                            center_translation=5)
        
        from data import extract_data_label
        x_test, y_test = extract_data_label(target_bags)
        x_test = x_test.to(device).float()

        # plot source data
        plt.figure(figsize=(5, 4))
        plt.scatter(source_loader.dataset.tensors[0][:, 0], source_loader.dataset.tensors[0][:, 1],
                    c=source_loader.dataset.tensors[1], cmap='viridis')

        plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap='viridis', marker='x')
    
    elif 1:
    
        from utils_local import loop_iterable

        config_file = './configs/office31.yaml'
        import yaml
        from data import get_office31
        cuda = True if torch.cuda.is_available() else False
        with open(config_file) as file:
            cfg = yaml.load(file, Loader=yaml.FullLoader)
        source = cfg['data']['files'][4][0]
        target = cfg['data']['files'][4][1]
        bag_size = cfg['data']['bag_size']
        nb_class_in_bag = cfg['data']['nb_class_in_bag']
        n_class = cfg['data']['n_class']
        dim = cfg['data']['dim']
        dim_latent = cfg['model']['dim_latent']
        n_hidden = cfg['model']['n_hidden']
        lr = cfg['bagCSI']['lr']
        num_epochs = cfg['bagCSI']['n_epochs']
        source_loader, target_bags = get_office31(source = source, target = target, batch_size=64, drop_last=True,
                    nb_missing_feat = None,
                    nb_class_in_bag = nb_class_in_bag,
                    bag_size = bag_size )
    elif 0:
        config_file = './configs/visda.yaml'
        import yaml
        from data import get_visda
        cuda = True if torch.cuda.is_available() else False
        with open(config_file) as file:
            cfg = yaml.load(file, Loader=yaml.FullLoader)
        bag_size = cfg['data']['bag_size']
        nb_class_in_bag = cfg['data']['nb_class_in_bag']
        n_class = cfg['data']['n_class']
        dim = cfg['data']['dim']
        dim_latent = cfg['model']['dim_latent']
        n_hidden = cfg['model']['n_hidden']
        num_epochs = 30
        classe_vec = [0,1,2,3,4,5,6,7,8,9,10,11]
        #classe_vec = [0,4,11]
        n_class = len(classe_vec)
        use_div = False
        source_loader, target_bags  = get_visda(batch_size=256, drop_last=True,
                    nb_class_in_bag = 10,
                    classe_vec=classe_vec,
                    bag_size = 50,
                    nb_missing_feat = None,
                    apply_miss_feature_source=False)

    

    from models import  FeatureExtractor, DataClassifier

    # Initialize the model, loss function, and optimizer        
    feat_extract = FeatureExtractor(input_dim=dim, n_hidden=n_hidden, output_dim=n_hidden)
    classifier = DataClassifier(input_dim=n_hidden, n_class=n_class)
    
    bagCSI_train(feat_extract,classifier, source_loader, target_bags, n_classes=n_class, num_epochs=num_epochs,device=device,
                    param_bag=1, param_da=1,verbose=True)

    
    
    from utils import evaluate_clf, create_data_loader, extract_data_label
    x_test, y_test = extract_data_label(target_bags, type_data='data', type_label='label')
    test_loader = create_data_loader(x_test, y_test, batch_size=128, shuffle=False,drop_last=False)
    model = nn.Sequential(feat_extract,classifier)
    model.to(device)
    acc, bal_acc, cm = evaluate_clf(model, test_loader,n_classes=n_class,return_pred=False)
    print(f'Accuracy: {bal_acc:.4f}')

#%%
































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
