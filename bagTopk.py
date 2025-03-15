
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
from models import FeatureExtractor, DataClassifier
from utils_local import loop_iterable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import ot
#from proportion_estimators import estimate_proportion
def dist_torch(x1,x2):
    x1p = x1.pow(2).sum(1).unsqueeze(1)
    x2p = x2.pow(2).sum(1).unsqueeze(1)
    prod_x1x2 = torch.mm(x1,x2.t())
    distance = x1p.expand_as(prod_x1x2) + x2p.t().expand_as(prod_x1x2) -2*prod_x1x2
    return distance 


def entropy_loss(v):
    """
    Entropy loss for probabilistic prediction vectors
    """
    return torch.mean(torch.sum(- torch.softmax(v, dim=1) * torch.log_softmax(v, dim=1), 1))





def bagTopK_train(feature_extractor,classifier_1, source_loader, target_bags, n_class, num_epochs=100,device='cpu',
                    lr=0.001,source_weight=1,
                    ent_weight=0.1, 
                    da_weight=0.,
                    topk=15,
                    mean_weight=0.1,
                    bag_weight=0.1,
                    verbose=False,large_source_loader=None):
    feature_extractor.train()
    classifier_1.train()
    #classifier_2.train()
    feature_extractor.to(device)
    classifier_1.to(device)
    #classifier_2.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer_feat = optim.Adam(feature_extractor.parameters(), lr=lr,betas=(0.9, 0.999))
    optimizer_c1 = optim.Adam(classifier_1.parameters(), lr=lr,betas=(0.9, 0.999))
    #optimizer_c2 = optim.Adam(classifier_2.parameters(), lr=lr,betas=(0.9, 0.999))
    
    for epoch in range(num_epochs):
        loss_1_epoch = 0
        loss_2_epoch = 0
        for i, (x_train, y_train) in enumerate(source_loader):
            x_train = x_train.to(device).float()
            y_train = y_train.to(device)
            i_bag = np.random.randint(0, len(target_bags))
            
            x_target = target_bags[i_bag]['data'].to(device).float()
            y_target_prop = torch.tensor(target_bags[i_bag]['prop']).to(device)

            
            # source loss
            source_feature = feature_extractor(x_train)
            outputs_1 = classifier_1(source_feature)
            loss_source_1 = criterion(outputs_1, y_train)

            # bag loss
            target_feature  = feature_extractor(x_target)
            outputs_target = classifier_1(target_feature)
            outputs_target = torch.softmax(outputs_target, dim=1)
            loss_bag = torch.mean(torch.abs(outputs_target.mean(dim=0) - y_target_prop))
            
            # IM loss 
            with torch.no_grad():
                outputs_target = classifier_1(target_feature)
            ent_loss = ent_weight * entropy_loss(outputs_target)
            msoftmax = nn.Softmax(dim=1)(outputs_target).mean(dim=0)
            ent_loss -= ent_weight* torch.sum(-msoftmax * torch.log(msoftmax + 1e-5) )

            # topk mean embedding loss
            arg_prop = torch.argsort(y_target_prop,descending=True)
            n_max = min(topk,len(arg_prop))
            loss_mean = 0
            for j in range(n_max):
                ind = arg_prop[j]
                source_feature_data = source_feature*(y_train==ind).float().view(-1,1)
                #print(source_feature_data.size())
                target_feature_data = target_feature*(y_target_prop[ind]).float().view(-1,1)
                #print(target_feature_data.size())
                loss_mean += torch.mean(torch.abs(source_feature_data.mean(dim=0) - target_feature_data.mean(dim=0)))**2
                
            # loss_mean = 0
            # for j in range(n_class):
            #     source_feature_data = source_feature*(y_train==j).float().view(-1,1)
            #     #print(source_feature_data.size())
            #     target_feature_data = target_feature*(y_target_prop[j]).float().view(-1,1)
            #     #print(target_feature_data.size())
            #     loss_mean += torch.mean(torch.abs(source_feature_data.mean(dim=0) - target_feature_data.mean(dim=0)))**2
                



            #  pseudo label
            if epoch > 0 and da_weight > 0:
                in_test = torch.where(y_target_prop > 0)[0]
                x_a, y_a = next(iter(large_source_loader))
                x_a = x_a.to(device).float()
                y_a = y_a.to(device)
                #print(y_a.shape)
                # keep all training data with the same label
                ind_tr = []
                for cls in  in_test:
                    ind = torch.where(y_a == cls)[0]
                    ind_tr.append(ind)
                ind_tr = torch.cat(ind_tr)
                source_feature_align = feature_extractor(x_a)
                source_feature_align = source_feature_align[ind_tr]
                y_a = y_a[ind_tr]


                n_1 = source_feature_align.size(0)  
                n_2 = target_feature.size(0)     
                a = torch.from_numpy(ot.unif(n_1)).float()
                b = torch.from_numpy(ot.unif(n_2)).float()
                y_target_prop = y_target_prop[in_test]  # keep only the classes in the target domain
                for i_c, cls in enumerate(in_test):
                    ind = torch.where(y_a  == cls)[0]
                    n_in_class = len(ind)          
                    a[ind] = y_target_prop[i_c]/n_in_class if n_in_class > 0 else 0.0
                b /= torch.sum(b)
                a /= torch.sum(a)
                b = b.float()
                a = a.float()
                if torch.sum(a) > 0 :
                    with torch.no_grad():
                        M = dist_torch(source_feature_align,target_feature)
                        gamma = ot.emd(a,b,M)
                        gamma = gamma.to(device)
                        ind_a = torch.argmax(gamma, dim=0)
                        y_t_pred = y_a[ind_a]
                        y_t = torch.Tensor(target_bags[i_bag]['label']).to(device).long()
                    bc = torch.sum(y_t_pred.eq(y_t))/len(y_t)
                    #print(bc)
                    output_target_ps = classifier_1(target_feature)
                    loss_ps = criterion(output_target_ps, y_t_pred)
            else:
                bc = 0
                loss_ps = 0.

            # optimizing feature extractor and classifier 1

            optimizer_feat.zero_grad()
            optimizer_c1.zero_grad()
            loss_1 = source_weight*loss_source_1 + bag_weight*loss_bag 
            loss_1 += ent_loss + da_weight*loss_ps + mean_weight*loss_mean
            loss_1.backward()
            optimizer_feat.step()
            optimizer_c1.step()
            loss_1_epoch += loss_1.item()



            #optimizing classifier 2
            # source_feature = feature_extractor(x_train)
            # outputs_2 = classifier_2(source_feature)
            # loss_source_2 = criterion(outputs_2, y_train)
            # optimizer_c2.zero_grad()
            # loss_2 = loss_source_2
            # loss_2.backward()
            # optimizer_c2.step()
            # loss_2_epoch += loss_2.item()




        loss_1_epoch /= len(source_loader)
        loss_2_epoch /= len(source_loader)
        if verbose:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss S+T: {loss_1_epoch:.4f} Loss S: {loss_2_epoch:.4f} bc: {bc:.4f}')





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
    
    elif 0:
    
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
        lr = cfg['bagMTL']['lr']
        num_epochs = cfg['bagMTL']['n_epochs']
        source_loader, target_bags = get_office31(source = source, target = target, batch_size=64, drop_last=True,
                    nb_missing_feat = None,
                    nb_class_in_bag = nb_class_in_bag,
                    bag_size = bag_size )
    elif 1:
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
        lr = cfg['bagTopk']['lr']
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


    x_train, y_train = source_loader.dataset.tensors
    large_source_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=1024, shuffle=True, drop_last=False)
    large_source_loader = loop_iterable(large_source_loader)
    input_size = x_train.size(1)
    # Initialize the model, loss function, and optimizer
    
    feat_extract = FeatureExtractor(input_dim=input_size, n_hidden=n_hidden, output_dim=n_hidden)
    classifier_1 = DataClassifier(input_dim=n_hidden, n_class=n_class)
    #classifier_2 = DataClassifier(input_dim=n_hidden, n_class=n_class)
    bagTopK_train(feat_extract,classifier_1, source_loader, target_bags, n_class=n_class, num_epochs=num_epochs,device=device,
                   source_weight=0.1,verbose=True, ent_weight=0.1,  da_weight=0.,
                   mean_weight=1,
                   bag_weight=1,
                   topk=1,
                   lr=lr,large_source_loader=large_source_loader)


    
    
    from utils import evaluate_clf, create_data_loader, extract_data_label

    x_test, y_test = extract_data_label(target_bags, type_data='data', type_label='label')
    test_loader = create_data_loader(x_test, y_test, batch_size=128, shuffle=False,drop_last=False)

    acc, bal_acc_1, cm = evaluate_clf(nn.Sequential(feat_extract,classifier_1), test_loader,n_classes=n_class,return_pred=False)
    #acc, bal_acc_2, cm = evaluate_clf(nn.Sequential(feat_extract,classifier_2), test_loader,n_classes=n_class,return_pred=False)
    print(f'Balanced Accuracy S+T: {bal_acc_1:.4f}')






    #%%
    # train_feat_mtl, train_label = extract_feature(nn.Sequential(feat_extract), source_loader, device=device)
    # test_feat_mtl, test_label = extract_feature(nn.Sequential(feat_extract), test_loader, device=device)
    # test_feat_mtl = target_bags[0]['data']
    # feat_extract.eval()
    # test_feat_mtl = feat_extract(test_feat_mtl)
    # y_target_prop = torch.tensor(target_bags[0]['prop']).to(device)
    # n_class = len(y_target_prop)
    # for j in range(n_class):
    #     source_feature_data = train_feat_mtl*(train_label==j).float().view(-1,1)
    #     print(source_feature_data.size())
    #     target_feature_data = test_feat_mtl*(y_target_prop[j]).float().view(-1,1)
    #     print(target_feature_data.size())
    #     loss_da = torch.mean(torch.abs(source_feature_data.mean(dim=0) - target_feature_data.mean(dim=0)))**2
        


    
    #%%
    
    
    
    
    
    data_mtl = torch.cat([train_feat_mtl, test_feat_mtl], dim=0)
    scaler = StandardScaler()
    data_mtl = scaler.fit_transform(data_mtl)
    z_mtl = TSNE(n_components=2,max_iter=100).fit_transform(data_mtl)
    x_a_mtl = z_mtl[:train_feat_mtl.size(0)]
    x_t_mtl = z_mtl[train_feat_mtl.size(0):]

    
    classe = [2,3,4]


    import matplotlib.pyplot as plt
    plt.figure(figsize=(5, 4))
    for i in classe:
        plt.scatter(x_a_mtl[train_label==i, 0], x_a_mtl[train_label==i, 1], label=f'class {i}',s=100)
        plt.scatter(x_t_mtl[test_label==i, 0], x_t_mtl[test_label==i,1], marker='x')
    plt.legend()
    plt.title('MTL')
    plt.show()# %%

# %%
