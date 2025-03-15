    #/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  10 11:08:03 2024

@author: alain
"""

#%%

import numpy as np
import os

import pandas as pd
import sys

# import matplotlib.pyplot as plt
# import seaborn as sns
# color_pal = sns.color_palette("colorblind", 11).as_hex()
# colors = ["cobalt blue","light orange", "neon pink", "cornflower","dark orange", "salmon pink"
#           ,"blue green", "aquamarine", "dark orange", "golden yellow", "reddish pink","black",  "reddish purple"]
# color_pal = sns.xkcd_palette(colors)
# plt.close("all")
import argparse

sys.argv=['']
parser = argparse.ArgumentParser(description="PropOT all expe ")
parser.add_argument("--s", type=int, default=0) 
args = parser.parse_args()


def fstr(template):
    return eval(f"f'{template}'")

def get_expe(filename):
    #filename = filename.replace('st_','st-')
    expe_ = filename.split('.npz')[0].split('-')
    expe = {}
    for a in range(0,len(expe_)-1,2):
        expe[expe_[a]] =expe_[a+1] 
    return expe


 

list_method = ['bagCSI','daLabelWD']
method_name = ['bagCSI','daLabelWD']
data_list =['office31','officehome','visda','mnist_usps','usps_mnist']



data_list.sort()

list_result = []    

for dirpath, dirnames, filenames in os.walk('./results/'):
    for filename in filenames:

        expe = get_expe(filename)
        try:
            algo = expe['algo']
        except:
            print(expe)
        dataset = expe['data']
        bag_size = int(expe['bag_size'])
        dep_sample = expe['dep_sample']
        st = int(expe['st'])
        nb_class_in_bag = int(expe['nb_class_in_bag'])
        if algo == 'daLabelWD':
            dist_loss_weight = float(expe['dist_loss_weight'])
            start_align = int(expe['start_align'])
            it = int(expe['iter_domain_classifier'])
        else:
            dist_loss_weight = 0
            start_align = 0
            it = 0
        
        df = np.load(os.path.join(dirpath, filename),allow_pickle=True)
        i_dataset = data_list.index(dataset)
        config = df['config']
        
        list_res_acc = df['list_bal_acc_test']
        non_nan_counts = np.sum(~np.isnan(list_res_acc))
        m_res_acc = np.nanmean(list_res_acc,axis=0)*100
        s_res_acc = np.nanstd(list_res_acc,axis=0)*100



        result = [dataset, st, algo, bag_size,nb_class_in_bag, start_align, it, dep_sample, dist_loss_weight, m_res_acc, s_res_acc, non_nan_counts]
        list_result.append(result)

list_tab = list_result.copy()
reference_list = ['office31','officehome','visda','mnist_usps','usps_mnist']

list_result.sort(key=lambda x: (reference_list.index(x[0]),x[1],x[2], x[5]))
pre_text = "dataset   n_per_class_anchor  n_neighbours "
dataset_old = ''
for i, result in enumerate(list_result):
    dataset, st, algo,bag_size, nb_class_in_bag, start_align, it, dep_sample, dist_loss_weight, m_res_acc, s_res_acc,nn_count = result

    if i==0 or dataset != dataset_old:
        print('-------------------------------------------------------')
        header = f"{'Data':10} ST {'Algorithm':12}   {'BS':4}   {'SA':4} {'it':2}  {'DistL':6}"
        print(header)
        print('-------------------------------------------------------')

        dataset_old = dataset

    texte = f"{dataset:10} {st:}  {algo:12} {bag_size:4} {start_align:} {it:}  {dist_loss_weight:7} "
    texte += f" {m_res_acc:2.2f} $\\pm$ {s_res_acc:2.2f} \t {nn_count:3}"
    print(texte)



list_tab.sort(key=lambda  x: (reference_list.index(x[0]),x[2],x[1], x[5]))









#%%







# def print_table_bag_size(list_tab, dataset_list, algo_list, bag_size_list, nb_class_in_bag_list,
#                          algo_name, data_name):

#     m_res_tab = np.zeros((len(dataset_list),len(bag_size_list),len(nb_class_in_bag_list),len(algo_list)))
#     s_res_tab = np.zeros((len(dataset_list),len(bag_size_list),len(nb_class_in_bag_list),len(algo_list)))
#     argmax_res_tab = np.zeros((len(dataset_list),len(bag_size_list),len(nb_class_in_bag_list),len(algo_list)))
#     for i, result in enumerate(list_tab):
#         dataset, algo,bag_size, n_per_class_anchor, nb_class_in_bag, dep_sample, m_res_acc, s_res_acc,nn_count = result
#         if dataset in dataset_list and (algo in algo_list) and (bag_size in bag_size_list) and nb_class_in_bag in nb_class_in_bag_list:

            
#             i_dataset = dataset_list.index(dataset)
#             i_bag_size = bag_size_list.index(bag_size)
#             i_nb_class_in_bag = nb_class_in_bag_list.index(nb_class_in_bag)
#             i_algo = algo_list.index(algo)
#             m_res_tab[i_dataset,i_bag_size,i_nb_class_in_bag,i_algo] = m_res_acc
#             s_res_tab[i_dataset,i_bag_size,i_nb_class_in_bag,i_algo] = s_res_acc
#             #i_max = np.argmax(m_res_acc,axis=3)

#     # table of maximum performance 
#     # removing full from maximum competition
#     if 'full' in algo_list:
#         i_algo = algo_list.index('full')
#         m_res_tab_copy = m_res_tab.copy()
#         m_res_tab_copy[:,:,:,i_algo] = 0
#     else:
#         m_res_tab_copy = m_res_tab.copy()


#     # table of maximum performance
#     for i_dataset,dataset in enumerate(dataset_list):
#         for i_bag_size,bag_size in enumerate(bag_size_list):
#             for i_nb_class_in_bag,nb_class_in_bag in enumerate(nb_class_in_bag_list):
#                 i_max = np.argmax(m_res_tab_copy[i_dataset,i_bag_size,i_nb_class_in_bag,:])
#                 i_max = np.argsort(m_res_tab_copy[i_dataset,i_bag_size,i_nb_class_in_bag,:])[-1]
#                 i_second = np.argsort(m_res_tab_copy[i_dataset,i_bag_size,i_nb_class_in_bag,:])[-2]
                
#                 argmax_res_tab[i_dataset,i_bag_size,i_nb_class_in_bag,i_max] = 1
#                 argmax_res_tab[i_dataset,i_bag_size,i_nb_class_in_bag,i_second] = 2





#     # print the table
#     for i_dataset,dataset in enumerate(dataset_list):
#         for i_algo,algo in enumerate(algo_list):
#             dataset = dataset.replace('_',' ')
#             algo = algo.replace('_',' ')
#             data_n = data_name[i_dataset]
#             algo_n = algo_name[i_algo]
#             if i_algo == 0:
#                 texte = f"{data_n:10} & {algo_n:15}"
#             else:
#                 texte = f"{'':10} & {algo_n:15}"
#             for i_bag_size, bag_size in enumerate(bag_size_list):
#                 for i_nb_class_in_bag,nb_class_in_bag in enumerate(nb_class_in_bag_list):
#                     m_res = m_res_tab[i_dataset,i_bag_size,i_nb_class_in_bag,i_algo]
#                     s_res = s_res_tab[i_dataset,i_bag_size,i_nb_class_in_bag,i_algo]
#                     if argmax_res_tab[i_dataset,i_bag_size,i_nb_class_in_bag,i_algo] == 1:
#                         texte += f" & \\textbf{{{m_res:2.2f} $\\pm$ {s_res:2.1f}}}"
#                     elif argmax_res_tab[i_dataset,i_bag_size,i_nb_class_in_bag,i_algo] == 2:
#                         texte += f" & \\underline{{{m_res:2.2f} $\\pm$ {s_res:2.1f}}}"
#                         argmax_res_tab[i_dataset,i_bag_size,i_nb_class_in_bag,i_algo] = 0
#                     else:
#                         texte += f" & {m_res:2.2f} $\\pm$ {s_res:2.1f}"
#                 if i_bag_size < len(bag_size_list)-1 and len(nb_class_in_bag_list)>1:
#                     texte += " &"
#             texte += " \\\\"
#             print(texte)

#     return m_res_tab, s_res_tab, argmax_res_tab


# #%%

# bag_size_list = [50,100,250]
# nb_class_in_bag_list = [2,0,100]

# summary_mean = []
# summary_ecart = []
# summary_win = []

# algo_list = ['full','daot','easyllp','propot','propot_laplace']
# algo_name = ['Full','DAOT','Easy LLP','LabelOT','LabelOT LapReg']

# dataset_list = ['dry_beans','har','indoor','optdigits','students','mnist','cifar']
# data_name = ['Dry Beans','HAR','Indoor','Optdigits','Students','MNIST','CIFAR']

# print('\n\n\n')
# print('------------ Multiclass Dep  Table -----------------')
# moy, eca, win = print_table_bag_size(list_tab, dataset_list, algo_list, bag_size_list, nb_class_in_bag_list,
#                      algo_name, data_name)

# summary_mean.append(moy)
# summary_ecart.append(eca)
# summary_win.append(win)

# dataset_list = ['adult','mushrooms','musk','phoneme','telescope','wine']
# data_name = ['Adult','Mushrooms','Musk','Phoneme','Telescope','Wine']
# num_bags_list = [10,50,250]
# num_bags_list_2 = [10,50,100]
# nb_class_in_bag_list = [2]

# print('\n\n\n ')
# print('------------ Binary Dep  Table -----------------')
# moy, eca, win = print_table_bag_size(list_tab, dataset_list, algo_list,bag_size_list, nb_class_in_bag_list,
#                      algo_name, data_name)

# summary_mean.append(moy)
# summary_ecart.append(eca)
# summary_win.append(win)


# dataset_list = ['dry_beans','har','indoor','optdigits','students','mnist','cifar']
# data_name = ['Dry Beans','HAR','Indoor','Optdigits','Students','MNIST','CIFAR']
# bag_size_list = [50,100,250]
# nb_class_in_bag_list = [10]
# print('\n\n\n')
# print('------------ Multiclass IID  Table -----------------')
# moy, eca, win = print_table_bag_size(list_tab,dataset_list, algo_list, bag_size_list, nb_class_in_bag_list,
#                      algo_name, data_name)

# summary_mean.append(moy)
# summary_ecart.append(eca)
# summary_win.append(win)

# dataset_list = ['adult','mushrooms','musk','phoneme','telescope','wine']
# data_name = ['Adult','Mushrooms','Musk','Phoneme','Telescope','Wine']

# print('\n\n\n ')
# print('------------ Binary IID  Table -----------------')
# moy, eca, win = print_table_bag_size(list_tab,dataset_list, algo_list, bag_size_list, nb_class_in_bag_list,
#                      algo_name, data_name)

# summary_mean.append(moy)
# summary_ecart.append(eca)
# summary_win.append(win)

# #%%
# meta_dataset = ['Multiclass Dep','Binary Dep','Multiclass IID','Binary IID']
# algo_list = ['Full','DAOT','Easy LLP','LabelOT','LabelOT LapReg']

# for i_algo,algo in enumerate(algo_list):
#     texte = f"{algo} "
#     for i_dataset,dataset in enumerate(meta_dataset):
#         m_moy = summary_mean[i_dataset].mean(axis=(0,1,2))[i_algo]
#         m_ecart = summary_ecart[i_dataset].mean(axis=(0,1,2))[i_algo]
#         m_win = int(summary_win[i_dataset].sum(axis=(0,1,2))[i_algo])
#         texte += f" & {m_moy:2.1f} $\\pm$ {m_ecart:2.0f} ({m_win:d})"
#     texte += " \\\\"
#     print(texte)






# #%%
# # --------------------------------------------------------------------------------------
# #
# #               semi supervised  
# #
# # --------------------------------------------------------------------------------------





# bag_size_list = [50,100,250]
# nb_class_in_bag_list = [2]

# summary_mean = []
# summary_ecart = []
# summary_win = []

# algo_list = ['full','propot','propot_laplace','propot_ssl','propot_ssl_laplace']
# algo_name = ['Full','LabelOT','LabelOT LapReg','LabelOT SSL','LabelOT SSL LapReg']

# algo_list = ['easyllp','propot','propot_laplace','propot_ssl','propot_ssl_laplace']
# algo_name = ['Easy LLP','LabelOT','LabelOT LapReg','LabelOT SSL','LabelOT SSL LapReg' ]


# dataset_list = ['adult','musk','mushrooms','wine','phoneme','telescope']
# data_name = ['adult','musk','mushrooms','wine','phoneme','telescope']


# print('\n\n\n')
# print('------------Semi-supervised -Binary Dep-----------------')
# moy, eca, win = print_table_bag_size(list_tab, dataset_list, algo_list, bag_size_list, nb_class_in_bag_list,
#                      algo_name, data_name)

# nb_class_in_bag_list = [10]
# print('\n\n\n')
# print('------------Semi-supervised -Binary Uniform-----------------')
# moy, eca, win = print_table_bag_size(list_tab, dataset_list, algo_list, bag_size_list, nb_class_in_bag_list,
#                      algo_name, data_name)





    
    

# %%
