import numpy as np
import matplotlib.pyplot as plt

import ot
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from sklearn.metrics import average_precision_score
from sklearn.metrics import balanced_accuracy_score
def create_data(X_train, y_train):
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    return train_dataset

def create_data_loader(X_train, y_train, batch_size=32, shuffle=True,drop_last=True):
    train_dataset = create_data(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return train_loader
def extract_data_label(Bag, type_label = 'label', type_data = 'data'):
    num_bags = len(Bag)
    
    X = torch.cat([Bag[i][type_data] for i in range(num_bags)],0)
    y = torch.cat([Bag[i][type_label] for i in range(num_bags)],0)
    if type_label == 'label' or type_label == 'y_pred':
        y = y.long()
    return X,y


def extract_feature(net,train_loader,device='cpu'):
    net.eval()
    train_feature = []
    train_label = []
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        label = targets.cpu()
        inputs, targets = inputs.to(device), targets.to(device)        

        outputs, feature = net(inputs,get_feature=True)
        feature = feature.detach().cpu()

        train_feature.append(feature)
        train_label.append(label)
                        
    train_label = torch.cat(train_label)
    train_feature = torch.cat(train_feature, 0)
    
    net.train()

    return train_feature, train_label

def evaluate_clf(clf, test_loader,n_classes=10,return_pred=False):
    clf.eval()  # Set the model to evaluation mode

    correct = 0
    total = 0
    all_correct = []
    all_predicted = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            test_outputs = clf(inputs.to(device)).cpu()
            _, predicted = torch.max(test_outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_correct.append(labels)
            all_predicted.append(predicted)
    accuracy = correct / total
    #print(f'Accuracy: {accuracy:.4f}')
    all_correct = torch.cat(all_correct)
    all_predicted = torch.cat(all_predicted)
    confusion_matrix = torch.zeros(n_classes,n_classes)
    b_accuracy = balanced_accuracy_score(all_correct, all_predicted)
    for t, p in zip(all_correct, all_predicted):
        confusion_matrix[t, p] += 1
    #print(confusion_matrix)
    clf.train()
    if return_pred:
        return accuracy, confusion_matrix, all_predicted
    else:
        return accuracy, b_accuracy, confusion_matrix

def error_on_bag_prop(model,val_bags,n_class=10):
    val_accur = []
    for bag in val_bags:
        data = bag['data']
        label = bag['label']
        prop = bag['prop']
        data = data.to(device).float()
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total = data.size(0)
        prop_pred = torch.bincount(predicted,minlength=n_class).float()/total
        acc = torch.sum(torch.abs(prop_pred - torch.Tensor(prop))).item()
        val_accur.append(acc)
    return torch.Tensor(val_accur).mean().item()
