
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
from torchvision import datasets, transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_bag_stats(Bag):
    bag_size = []
    bag_spread = []
    for bag in Bag:
        bag_size.append(bag['data'].shape[0])
        bag_sp = np.array(bag['prop']).max() - np.array(bag['prop']).min()
        bag_spread.append(bag_sp)
    bag_size = np.array(bag_size)
    bag_spread = np.array(bag_spread)
    #print(bag_size)
    stats = {'min':bag_size.min(), 'mean':bag_size.mean(), 'max':bag_size.max(),'len':len(Bag),
             'min_spread':bag_spread.min(),'mean_spread':bag_spread.mean(),'max_spread':bag_spread.max(),
             'bag_size':bag_size, 'bag_spread':bag_spread}
    return stats


# Function to extract embeddings
def extract_embeddings(data_loader):

    # Load pre-trained ResNet model
    model = models.resnet18(pretrained=True)

    # Modify the model to output embeddings (remove the final layer)
    model = nn.Sequential(*(list(model.children())[:-1]))  # Remove the last fully connected layer
    model = model.to(device)
    model.eval()  # Set the model to evaluation mode


    embeddings = []
    emb_labels = []
    data = []
    with torch.no_grad():  # Disable gradient calculation
        for inputs, labels in data_loader:
            data.append(inputs.cpu())
            inputs = inputs.to(device)
            output = model(inputs)  # Get the output from the modified model
            output = output.view(output.size(0), -1)  # Flatten the output
            embeddings.append(output.cpu())  # Move to CPU and store
            emb_labels.append(labels)
    return torch.cat(embeddings), torch.cat(data), torch.cat(emb_labels)  # Concatenate all embeddings


def normal_sample(n, dim, centers = [(0, 0)], scale=1, var=1):
    m = torch.distributions.multivariate_normal.MultivariateNormal(
        torch.zeros(dim), math.sqrt(var) * torch.eye(dim)
    )
    data = torch.Tensor(centers) * scale + m.sample((n,))
    return data

def add_noise_to_prop(Bag,scale=0.1):
    for bag in Bag:
        prop = bag['prop']
        n_class = len(prop)
        good_prop = False
        while not good_prop:
            lap = np.random.laplace(0,scale,n_class)
            prop_n = np.array(prop) + lap
            prop_n = np.clip(prop_n,0,1)
            good_prop = np.sum(prop_n) > 0
        prop_n = prop_n/prop_n.sum()
        bag['prop'] = list(prop_n)
    return Bag



def create_test_data(Bag, n_s = 1000, variance = 1, dim = 20):
    centers = Bag[0]['centers']

    x = []
    y = []
    for i in range(len(centers)):
        # len (centers) is the number of class
        # len(centers[i]) is the number of mode in the class
        n_mode  =  n_s // len(centers[i])//len(centers)
        for j in range(len(centers[i])):
            current_center = centers[i][j]
            xaux = normal_sample(n_mode, dim , current_center , scale=1, var=variance)
            yaux = torch.ones(n_mode)*i
            x.append(xaux)
            y.append(yaux)

    x = torch.cat(x)
    y = torch.cat(y)
    return x,y


def make_multi_blobs(n_train, centers, sigma_vec=0.01, dim=2):
    """
    Generate multi-blob data.

    Parameters:
    n_train (list of int): Number of samples for each blob.
    centers (list of arrays): Centers of the blobs.
    sigma_vec (float): Standard deviation for the blobs.
    dim (int): Dimensionality of the data.

    Returns:
    X (ndarray): Generated data points.
    y (ndarray): Labels for the data points.
    """
    X = np.zeros((0, dim))
    y = np.zeros(0)

    for i in range(len(centers)):
        # Create a covariance matrix with sigma_vec on the diagonal
        cov = np.eye(dim) * sigma_vec
        # Generate data points for the current blob
        aux = np.random.multivariate_normal(centers[i], cov, n_train[i])
        # Stack the generated data points
        X = np.vstack((X, aux))
        # Stack the labels
        y = np.hstack((y, np.ones(n_train[i]) * i))

    return torch.from_numpy(X), torch.from_numpy(y).long()

def extract_data_label(Bag, type_label = 'label', type_data = 'data'):
    num_bags = len(Bag)
    
    X = torch.cat([Bag[i][type_data] for i in range(num_bags)],0)
    y = torch.cat([Bag[i][type_label] for i in range(num_bags)],0)
    if type_label == 'label' or type_label == 'y_pred':
        y = y.long()
    return X,y



def create_bags_from_data_iid(train_data, train_labels,bag_size=100,embeddings=None):
    # create bags from the data with a fixed size
    # input: train_data, train_labels, bag_size
    # 
    # 
    # output: Bag
    if isinstance(train_data,np.ndarray):
        train_data = torch.from_numpy(train_data).float()
        train_labels = torch.from_numpy(train_labels).long()
    n_class = len(torch.unique(train_labels))
    n_data = len(train_labels)
    Bag = []
    n_batch = n_data // bag_size
    ind_perm = np.random.permutation(n_data)
    for i in range(n_batch):
        ind = ind_perm[i*bag_size:(i+1)*bag_size]
        Bag.append({'ind':ind, 'data':train_data[ind],'label':train_labels[ind],
                     'embeddings':embeddings['train'][ind] if embeddings is not None else None,
                    'prop':[torch.sum(train_labels[ind]==i_cls).item()/len(ind) for i_cls in range(n_class)]})

    return Bag



def create_bags_from_data_dep(train_data, train_labels, bag_size, nb_class_in_bag,embeddings=None, max_sample_per_bag=1e6):
 
    if isinstance(train_data,np.ndarray):
        train_data = torch.from_numpy(train_data).float()
        train_labels = torch.from_numpy(train_labels).long()

    n_class = len(torch.unique(train_labels))
    num_bags = train_data.shape[0]//bag_size
    print(n_class)


    good_bag = False
    while not good_bag:
        Bag = []
        nb_time_class_is_inbag = np.zeros(n_class)
        for i in range(num_bags):
            class_in_bag = np.random.choice(n_class,nb_class_in_bag,replace=False)
            Bag.append({'class':class_in_bag})
            nb_time_class_is_inbag[class_in_bag] += 1
        if np.all(nb_time_class_is_inbag>0):
            good_bag = True

    print('Repartition of class accross bag', nb_time_class_is_inbag)

    # separating each class of the dataset into subsets according to the number of time they are in  bags
    class_for_bag = []
    for i_cls in range(n_class):
        ind = torch.where(train_labels==i_cls)[0].numpy()
        n_i_class = len(ind)

        permuted_array = np.random.permutation(ind)
        nb_subset = nb_time_class_is_inbag[i_cls].astype(int)
        # Step 2: Split the permuted array into k subsets
        ind_subset = np.sort(np.random.choice(n_i_class, nb_subset - 1, replace=False))

        start_subset =  np.concatenate(([0], ind_subset))
        end_subset = np.concatenate((ind_subset, [n_i_class]))
        subsets = [permuted_array[start_subset[i]:end_subset[i]] for i in range(nb_subset)]
        class_for_bag.append(subsets)
        #print(i_cls,subsets)
    # assign the data to the bag
    for i_bag in range(num_bags):
        class_in_bag = Bag[i_bag]['class']
        ind_for_bag = []
        for i_cls in class_in_bag:
            if len(class_for_bag[i_cls]) == 0:
                print('Error: not enough data for class',i_cls)
            ind_class = class_for_bag[i_cls].pop()
            Bag[i_bag][i_cls] = ind_class
            ind_for_bag.extend(ind_class)

        if len(ind_for_bag) > max_sample_per_bag:
            ind_for_bag = np.random.choice(ind_for_bag, max_sample_per_bag, replace=False)

        Bag[i_bag]['ind'] = ind_for_bag
        Bag[i_bag]['data'] = train_data[ind_for_bag]
        Bag[i_bag]['label'] = train_labels[ind_for_bag]
        if embeddings is not None:
            Bag[i_bag]['embeddings'] = embeddings['train'][ind_for_bag]


        # computing propotion of each class in the bag
        #print(i_bag,class_in_bag,ind_for_bag)
        prop = [torch.sum(train_labels[ind_for_bag]==i_cls).item()/len(ind_for_bag) for i_cls in range(n_class)]
        Bag[i_bag]['prop'] = prop

    return Bag

def get_usps_mnist(batch_size=32, drop_last=True,
                        nb_missing_feat = 10,
                        nb_class_in_bag = 10,
                        dep_sample = 1,
                        bag_size = 50,
                        path = './data/',
                        miss_feature=None,
                        apply_miss_feature_target=False,
                        apply_miss_feature_source=False):

    
    # Define the transformations for the dataset
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load the USPS dataset

    train_dataset = datasets.USPS(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False, 
        drop_last=False)
    data,label = next(iter(train_loader))
    full_data = torch.utils.data.TensorDataset(data.float(), label.long())

    source_loader = torch.utils.data.DataLoader(
        dataset=full_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)



    # transform = transforms.Compose([
    # transforms.Resize((28, 28)),
    # transforms.ToTensor(),
    # transforms.Normalize((0.1307,), (0.3081,))
    # ])

    # Load the MNIST dataset
    #train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, 
        drop_last=False)
    data,label = next(iter(test_loader))
    if dep_sample == 1:
        target_bags = create_bags_from_data_dep(data, label, bag_size, nb_class_in_bag,embeddings=None, max_sample_per_bag=1e6)
    else:
        target_bags = create_bags_from_data_iid(data, label, bag_size,embeddings
                                                =None)

    return source_loader, target_bags


def get_mnist_usps(batch_size=32, drop_last=True,
                        nb_missing_feat = 10,
                        nb_class_in_bag = 10,
                        bag_size = 50,
                        dep_sample = 1,
                        path = './data/',
                        miss_feature=None,
                        apply_miss_feature_target=False,
                        apply_miss_feature_source=False):

    
    # Define the transformations for the dataset
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])


    # Load the MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False, 
        drop_last=False)
    data,label = next(iter(train_loader))
    full_data = torch.utils.data.TensorDataset(data.float(), label.long())

    source_loader = torch.utils.data.DataLoader(
        dataset=full_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)

    # Load the USPS dataset

    test_dataset = datasets.USPS(root='./data', train=True, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, 
        drop_last=False)
    data,label = next(iter(test_loader))
    if dep_sample == 1:
        target_bags = create_bags_from_data_dep(data, label, bag_size, nb_class_in_bag,embeddings=None, max_sample_per_bag=1e6)
    else:
        target_bags = create_bags_from_data_iid(data, label, bag_size,embeddings
                                                =None)
    return source_loader, target_bags



def get_officehome(source = 'Art_Art', target = 'Art_Clipart', batch_size=32, drop_last=True,
                         nb_missing_feat = 10,
                        nb_class_in_bag = 10,
                        bag_size = 50,
                        dep_sample = 1,
                        path = './data/office/',
                        miss_feature=None,
                        apply_miss_feature_target=False,
                        apply_miss_feature_source=False): 
    

    data_dic = np.load(path + 'officehome.npy', allow_pickle=True)
    data = torch.from_numpy(data_dic.item()[source][0]).float()
    label = torch.from_numpy(data_dic.item()[source ][1]).long()
    if miss_feature == None and apply_miss_feature_target:
        miss_feature = np.random.choice(data.shape[1],nb_missing_feat,replace=False)
        mask = torch.ones(data.size(1), dtype=torch.bool)
        mask[miss_feature] = False


    if apply_miss_feature_source:
        full_data = torch.utils.data.TensorDataset(data[:,mask], label.long())
    else:
        full_data = torch.utils.data.TensorDataset(data, label.long())
    source_loader = torch.utils.data.DataLoader(
        dataset=full_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last)



    data_dic = np.load(path + 'officehome.npy', allow_pickle=True)
    data = torch.from_numpy(data_dic.item()[target][0]).float()
    label = torch.from_numpy(data_dic.item()[target][1]).long()
    if apply_miss_feature_target:
        data = data[:,mask].float()
    if dep_sample == 1:
        target_bags = create_bags_from_data_dep(data, label, bag_size, nb_class_in_bag,embeddings=None, max_sample_per_bag=1e6)
    else:
        target_bags = create_bags_from_data_iid(data, label, bag_size,embeddings
                                                =None)

    return source_loader, target_bags


def get_office31(source = 'amazon_amazon', target = 'amazon_dslr', batch_size=32, drop_last=True,
                         nb_missing_feat = 10,
                         nb_class_in_bag = 5,
                        bag_size = 50,
                        dep_sample = 1,
                        path = './data/office/office31_resnet50/',
                        miss_feature=None,
                        apply_miss_feature_source=False,
                        apply_miss_feature_target=False,
                        ): 
    

    

    df = pd.read_csv(path + source + '.csv')  
    data = torch.from_numpy(df.values[:,0:2048]).float()
    label = torch.from_numpy(df.values[:,2048]).long()
    if miss_feature == None:
        miss_feature = np.random.choice(data.shape[1],nb_missing_feat,replace=False)
        mask = torch.ones(data.size(1), dtype=torch.bool)
        mask[miss_feature] = False
                         

    if apply_miss_feature_source:
        full_data = torch.utils.data.TensorDataset(data[:,mask], label.long())
    else:
        full_data = torch.utils.data.TensorDataset(data, label.long())
    source_loader = torch.utils.data.DataLoader(
        dataset=full_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last)



    df = pd.read_csv(path + target + '.csv')  
    

    data = torch.from_numpy(df.values[:,0:2048]).float()
    label = torch.from_numpy(df.values[:,2048]).long()
    if apply_miss_feature_target:
        data = data[:,mask].float()
    if dep_sample == 1:
        target_bags = create_bags_from_data_dep(data, label, bag_size, nb_class_in_bag,embeddings=None, max_sample_per_bag=1e6)
    else:
        target_bags = create_bags_from_data_iid(data, label, bag_size,embeddings
                                                =None)
    return source_loader, target_bags




def get_visda(batch_size=32, drop_last=True,
                        nb_missing_feat = 10,
                        nb_class_in_bag = 10,
                        dep_sample = 1,
                        bag_size = 50,
                        classe_vec=[0,4,11],
                        path = './data/visda/',
                        miss_feature=None,
                        apply_miss_feature_source=False,
                        apply_miss_feature_target=False,): 
    
                             
                        


    aux = [str(i) for i in classe_vec]
    if len(classe_vec) < nb_class_in_bag:
        nb_class_in_bag = len(classe_vec)
    train = True
    filename = 'visda-train'+ ''.join(aux)+'.npz'
    res = np.load(path+filename)
    data = torch.from_numpy(res['X']).float()
    label = torch.from_numpy(res['y']).long()
    if miss_feature == None:
        miss_feature = np.random.choice(data.shape[1],nb_missing_feat,replace=False)
        mask = torch.ones(data.size(1), dtype=torch.bool)
        mask[miss_feature] = False


    if apply_miss_feature_source:
        full_data = torch.utils.data.TensorDataset(data[:,mask], label.long())
    else:
        full_data = torch.utils.data.TensorDataset(data, label.long())
    source_loader = torch.utils.data.DataLoader(
        dataset=full_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last)


    filename = 'visda-val'+ ''.join(aux)+'.npz'
    res = np.load(path+filename)
    data = torch.from_numpy(res['X']).float()
    label = torch.from_numpy(res['y']).long()
    if apply_miss_feature_target:
        data = data[:,mask] 
    if dep_sample == 1:
        target_bags = create_bags_from_data_dep(data, label, bag_size, nb_class_in_bag,embeddings=None, max_sample_per_bag=1e6)
    else:
        target_bags = create_bags_from_data_iid(data, label, bag_size,embeddings
                                                =None)
    return source_loader, target_bags

def get_toy(batch_size=32, drop_last=True,
                        nb_missing_feat = 10,
                        nb_class_in_bag = 3,
                        bag_size = 50,
                        dim = 2,
                        n_centers = 3,
                        n_train = [300,300,300],
                        n_test = None,
                        data_variance = 0.01,
                        center_variance = 10,
                        center_translation = 3,
                        miss_feature=None,
                        apply_miss_feature_source=False): 
    

    if dim == 2 or miss_feature == None:
        nb_missing_feat = 0
    elif nb_missing_feat > dim:
        nb_missing_feat = dim//4         
    
    centers = np.random.multivariate_normal(np.zeros(dim), np.eye(dim) * center_variance, n_centers)
    center_translation = np.random.multivariate_normal(np.zeros(dim), np.eye(dim) * center_translation, 1)
    X, y = make_multi_blobs(n_train, centers, sigma_vec=data_variance, dim=dim)
    X = X.float()/5
    if n_test == None:
        n_test = [ n_i*10 for n_i in n_train]
    Xt,yt = make_multi_blobs(n_test, centers + center_translation, sigma_vec=data_variance, dim=dim)
    Xt = Xt.float()/5
    full_data = torch.utils.data.TensorDataset(X, y)
    if miss_feature == None:
        miss_feature = np.random.choice(X.shape[1],nb_missing_feat,replace=False)
        mask = torch.ones(X.size(1), dtype=torch.bool)
        mask[miss_feature] = False
    if apply_miss_feature_source:
        full_data = torch.utils.data.TensorDataset(X[:,mask].float(), y.long())
    else:
        full_data = torch.utils.data.TensorDataset(X.float(), y.long())

    source_loader = torch.utils.data.DataLoader(
        dataset=full_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last)


    Xt = Xt[:,mask].float() 
    target_bags = create_bags_from_data_dep(Xt, yt, bag_size, nb_class_in_bag,embeddings=None, max_sample_per_bag=1e6)

    return source_loader, target_bags


if __name__ == "__main__":


    if  0:
        source = 'amazon_amazon'
        target = 'amazon_dslr'
        source_loader, target_bags = get_office31(source=source, target=target,bag_size=50,apply_miss_feature_target=True)

    if  0:
        source_loader, target_bags = get_visda(apply_miss_feature_source=True,bag_size=50)
    if  0:

        source_loader, target_bags = get_toy(apply_miss_feature_source=True,bag_size=50)
    if 0:
        source_loader, target_bags = get_officehome(source = 'Art_Art', target = 'Art_Clipart',
                                                    apply_miss_feature_target=True,
                                                    bag_size=50)

    if 1:
        source_loader, target_bags = get_usps_mnist(dep_sample=1,bag_size=1000)
    if 0:
        source_loader, target_bags = get_mnist_usps()
    print(len(source_loader.dataset),len(target_bags))


    #%%

    #%%




    # print('source_loader',source_loader.dataset.tensors[0].shape)
    # print('target_bags',target_bags[0]['data'].shape)
    # s = 0
    # for bag in target_bags:
    #     s += bag['data'].shape[0]
    # print(s/len(target_bags))






    #%%
    # check the config file
    if 1:   
        import yaml
        config_file = './configs/officehome.yaml'
        with open(config_file) as file:
            cfg = yaml.load(file, Loader=yaml.FullLoader)
        for (source,target) in cfg['data']['files']:
            source_loader, target_bags = get_officehome(source = source, target = target,apply_miss_feature_target=True)

            print('source_loader', source, source_loader.dataset.tensors[0].shape)
            print('target_bags',target, target_bags[0]['data'].shape)
    if 1:   
            import yaml
            config_file = './configs/office31.yaml'
            with open(config_file) as file:
                cfg = yaml.load(file, Loader=yaml.FullLoader)
            for (source,target) in cfg['data']['files']:
                source_loader, target_bags = get_office31(source = source, target = target,apply_miss_feature_target=True)

                print('source_loader', source, source_loader.dataset.tensors[0].shape)
                print('target_bags',target, target_bags[0]['data'].shape)




    # %%
    path = './data/office/'
    data_dic = np.load(path + 'officehome.npy', allow_pickle=True)
    len(data_dic.item().keys())

    # %%
