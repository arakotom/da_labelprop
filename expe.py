
#%%
import math
import os
import time

import numpy as np
import ot 
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import argparse
import warnings
from bagCSI import bagCSI_train
from models import FullyConnectedNN
from data import get_toy, get_visda, get_officehome, get_office31
from utils import create_data_loader, evaluate_clf, extract_data_label, error_on_bag_prop
from utils_local import evaluate_data_classifier
import yaml 
import sys
warnings.filterwarnings("ignore", category=UserWarning) 
if __name__ == '__main__':


    
    sys.argv = ['']
    args = argparse.Namespace()

    parser = argparse.ArgumentParser(description='training llp models')

    # general parameters
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--expe_name', type=str,default="")
    parser.add_argument('--nb_iter', type=int, default=10)
    parser.add_argument('--data', type=str, default='officehome')
    parser.add_argument('--algo', type=str, default='bagCSI')
    args = parser.parse_args()
    config_file = f"./configs/{args.data}.yaml"
    with open(config_file) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
    print(cfg)

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = args.data
    algo = args.algo
    dep_sample = cfg['data']['dep_sample']  
    nb_class_in_bag = cfg['data']['n_class']
    bag_size = cfg['data']['bag_size']

    print(args.expe_name,data)

    list_acc_test = []
    list_bal_acc_test = []
    

    for iter in range(9,args.nb_iter):
        np.random.seed(seed + iter )
        torch.manual_seed(seed + iter)
        torch.cuda.manual_seed_all(seed + iter)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if data == 'toy':
            dim = cfg['data']['dim']
            variance = cfg['data']['variance']
            n_class = cfg['data']['n_class']
            dim_latent = cfg['model']['dim_latent']
            n_hidden = cfg['model']['n_hidden']
            use_div = False 
            source_loader, target_bags = get_toy(apply_miss_feature_source=False,
                                                dim=dim,
                                                data_variance=variance,
                                                center_translation=10,
                                                nb_class_in_bag= cfg['data']['n_class'],
                                                bag_size=cfg['data']['bag_size'],)
            if args.expe_name == "":
                savedir = 'results/toy'
            else:
                savedir = 'results/toy-' + args.expe_name

        if data == 'visda':
            n_class = 12
            dim = 2048
            dim_latent = 512
            n_hidden = 512
            use_div = False
            source_loader, target_bags  = get_visda(batch_size=128, drop_last=True,
                        nb_class_in_bag = 10,
                        bag_size = 50,
                        apply_miss_feature_source=False)
            if args.expe_name == "":
                savedir = 'results/visda'
            else:
                savedir = 'results/visda-' + args.expe_name
        if data == 'officehome':
            n_class = 65
            dim = 2048
            dim_latent = 512
            n_hidden = 512
            use_div = True
            source_loader, target_bags = get_officehome(source = 'Art_Art', target = 'Art_Clipart', batch_size=128, drop_last=True,
                         nb_missing_feat = None,
                        nb_class_in_bag = 10,
                        bag_size = 50)

            if args.expe_name == "":
                savedir = 'results/officehome'
            else:
                savedir = 'results/officehome-' + args.expe_name
        
        if data == 'office31':
            n_class = 31
            dim = 2048
            dim_latent = 512
            n_hidden = 512
            use_div = True
            source_loader, target_bags = get_office31(source = "amazon_amazon", target = "amazon_dslr", batch_size=128, drop_last=True,
                        nb_missing_feat = None,
                        nb_class_in_bag = 10,
                        bag_size = 50)

            if args.expe_name == "":
                savedir = 'results/office31'
            else:
                savedir = 'results/office31-' + args.expe_name


        nb_val = len(target_bags)//10
        val_bags = target_bags[:nb_val]
        target_bags = target_bags[nb_val:]
      
        X_test, y_test = extract_data_label(target_bags, type_data='data', type_label='label')
        test_loader = create_data_loader(X_test, y_test, batch_size=128, shuffle=False,drop_last=False)

        filesave = f"data-{data}-algo-{algo}-dep_sample-{dep_sample}-nb_class_in_bag-{nb_class_in_bag}-bag_size-{bag_size}"
         

        if args.algo == 'bagCSI':
            pass
        if args.algo == 'daLabelOT':
            pass



        if args.data == 'toy':
            filesave += f"-dim-{dim}-variance-{variance}"
        filesave += f'-seed-{seed}'
        os.makedirs(savedir, exist_ok=True)




    

        acc_test = 0
        bal_acc_test = 0



        #%%
        #----------------------------------------------------------------------------------------
        #       bagCSI
        #----------------------------------------------------------------------------------------

        if args.algo == 'bagCSI':
            #%%
            print('bagCSI')

            input_size = dim
            learning_rate = cfg['bagCSI']['lr']
            num_epochs = cfg['bagCSI']['n_epochs']
            val_max = n_class

            for param_bag in [10]:
                for param_da in [0]:
                    model = FullyConnectedNN(input_size, n_hidden= n_hidden, n_class=n_class)
                    bagCSI_train(model, source_loader, target_bags, n_classes=n_class, num_epochs=num_epochs,device=device,
                                    param_bag=param_bag, param_da=param_da,
                                    learning_rate=learning_rate,
                                    verbose=True)
                    model.eval()
                    val_err = error_on_bag_prop(model,val_bags,n_class=n_class)
                    if val_err < val_max:
                        val_max = val_err
                        best_param_bag = param_bag
                        best_param_da = param_da
                        acc_test, bal_acc_test, cm_test = evaluate_clf(model, test_loader,n_classes=n_class)

                    print(f'bag {param_bag} da {param_da} {val_max:.4f} {bal_acc_test:.4f}')




            #%%
        if args.algo == 'daLabelOTPhi':
            #%%
            print('DALabelOTPhi')
            from daLabelOT import daLabelOT
            from models import ResidualPhi, DomainClassifier, DataClassifier, FeatureExtractor
            from utils_local import set_optimizer_data_classifier, set_optimizer_data_classifier_t, set_optimizer_domain_classifier, set_optimizer_feat_extractor, set_optimizer_phi
            from utils_local import estimate_source_proportion


            cuda = True if torch.cuda.is_available() else False
            with open(config_file) as file:
                cfg = yaml.load(file, Loader=yaml.FullLoader)

            ent_weight = cfg['daLabelOT']['ent_weight']
            clf_t_weight = cfg['daLabelOT']['clf_t_weight']
            div_weight = cfg['daLabelOT']['div_weight']
            n_epochs =  cfg['daLabelOT']['n_epochs']            # total number of epochs
            epoch_start_g = cfg['daLabelOT']['epoch_start_g'] #args.epoch_start_g  # epoch to start retrain the feature extractor
            lr = cfg['daLabelOT']['lr']
            start_align = cfg['daLabelOT']['start_align']
            use_div = cfg['daLabelOT']['use_div']
            nblocks = cfg['daLabelOT']['nblocks']
            proportion_S = estimate_source_proportion(source_loader, n_clusters=n_class)
            val_max = n_class
            it = iter
            for bag_loss_weight in [10]:

                feat_extract_dalabelot = FeatureExtractor(dim, n_hidden=n_hidden, output_dim=dim_latent)
                data_class_dalabelot = DataClassifier(input_dim= dim_latent, n_class=n_class)
                data_class_t_dalabelot = DataClassifier(input_dim= dim_latent, n_class=n_class)
                phi_dalabelot = ResidualPhi(nblocks=nblocks, dim= dim_latent, nl_layer='relu', norm_layer='batch1d', n_branches=1)
                dalabelot = daLabelOT(feat_extract_dalabelot, data_class_dalabelot, phi_dalabelot, source_loader, target_bags,
                                    cuda=cuda,
                                n_class=n_class, 
                                epoch_start_align=start_align, init_lr=lr,
                                use_div=use_div, n_epochs=n_epochs,
                                clf_t_weight=clf_t_weight, iter=it, epoch_start_g=epoch_start_g,
                                div_weight=div_weight,
                                data_class_t=data_class_t_dalabelot, ent_weight=ent_weight,proportion_S=proportion_S,
                                bag_loss_weight=bag_loss_weight)
                set_optimizer_phi(dalabelot, optim.Adam(dalabelot.phi.parameters(), lr=lr, betas=(0.5, 0.999)))
                set_optimizer_data_classifier_t(dalabelot, optim.Adam(dalabelot.data_classifier_t.parameters(), lr=lr, betas=(0.5, 0.999)))
                set_optimizer_feat_extractor(dalabelot, optim.Adam(dalabelot.feat_extractor.parameters(), lr=0, betas=(0.5, 0.999)))
                set_optimizer_data_classifier(dalabelot, optim.Adam(dalabelot.data_classifier.parameters(), lr=lr*10, betas=(0.5, 0.999)))
                dalabelot.fit()
                #acc_test, bal_acc_test = evaluate_data_classifier(dalabelot, test_loader, is_target=True, is_ft=True)
                #print(f' {bag_loss_weight}  {bal_acc_test:.4f}')
                #%%
                model_s = nn.Sequential(feat_extract_dalabelot,  data_class_dalabelot)
                model_s.eval()
                acc_test, bal_acc_s, cm_test = evaluate_clf(model_s, source_loader,n_classes=n_class)
                print(f' {bag_loss_weight}  {bal_acc_s:.4f}')
            
                #%%
                model = nn.Sequential(feat_extract_dalabelot,  data_class_t_dalabelot)



                model.eval()
                val_err = error_on_bag_prop(model,val_bags,n_class=n_class)
                if val_err < val_max:
                    val_max = val_err
                    best_bag_loss_weight = bag_loss_weight
                    acc_test, bal_acc_test = evaluate_data_classifier(dalabelot, test_loader, is_target=True, is_ft=True)
            
                print(f' {bag_loss_weight}  {val_max:.4f} {bal_acc_test:.4f}')

            #%%
            print("Accuracy on the test set: ", bal_acc_test)
            #%%
       
        if args.algo == 'daLabelWD':
            #%%
            print('DALabelWD')
            from daLabelWD import daLabelWD
            from models import DomainClassifier, DataClassifier, FeatureExtractor
            from utils_local import set_optimizer_data_classifier, set_optimizer_domain_classifier, set_optimizer_feat_extractor, set_optimizer_phi
            from utils_local import estimate_source_proportion


            cuda = True if torch.cuda.is_available() else False
            with open(config_file) as file:
                cfg = yaml.load(file, Loader=yaml.FullLoader)

            ent_weight = cfg['daLabelOT']['ent_weight']
            clf_t_weight = cfg['daLabelOT']['clf_t_weight']
            div_weight = cfg['daLabelOT']['div_weight']
            n_epochs =  cfg['daLabelOT']['n_epochs']            # total number of epochs
            epoch_start_g = cfg['daLabelOT']['epoch_start_g'] #args.epoch_start_g  # epoch to start retrain the feature extractor
            lr = cfg['daLabelOT']['lr']
            start_align = cfg['daLabelOT']['start_align']
            use_div = cfg['daLabelOT']['use_div']
            nblocks = cfg['daLabelOT']['nblocks']
            proportion_S = estimate_source_proportion(source_loader, n_clusters=n_class)
            val_max = n_class
            it = iter
            for bag_loss_weight in [10]:

                feat_extract_dalabelot = FeatureExtractor(dim, n_hidden=n_hidden, output_dim=dim_latent)
                data_class_dalabelot = DataClassifier(input_dim= dim_latent, n_class=n_class)
                domain_class_dalabelot = DomainClassifier(input_dim= dim_latent,n_hidden=n_hidden)

                
                dalabelot = daLabelWD(feat_extract_dalabelot, data_class_dalabelot, domain_class_dalabelot, source_loader, target_bags,
                                    cuda=cuda,
                                n_class=n_class, 
                                epoch_start_align=start_align, init_lr=lr,
                                #use_div=use_div,
                                n_epochs=n_epochs,
                                #clf_t_weight=clf_t_weight, 
                                iter=it, 
                                epoch_start_g=epoch_start_g,
                                iter_domain_classifier=1,
                                #div_weight=div_weight,
                                #data_class_t=data_class_t_dalabelot, ent_weight=ent_weight,
                                proportion_S=proportion_S,
                                bag_loss_weight=bag_loss_weight)
                set_optimizer_feat_extractor(dalabelot, optim.Adam(dalabelot.feat_extractor.parameters(), lr=lr, betas=(0.5, 0.999)))
                set_optimizer_data_classifier(dalabelot, optim.Adam(dalabelot.data_classifier.parameters(), lr=lr, betas=(0.5, 0.999)))
                set_optimizer_domain_classifier(dalabelot, optim.Adam(dalabelot.domain_classifier.parameters(), lr=lr, betas=(0.5, 0.999)))
                dalabelot.fit()
                #acc_test, bal_acc_test = evaluate_data_classifier(dalabelot, test_loader, is_target=True, is_ft=True)
                #print(f' {bag_loss_weight}  {bal_acc_test:.4f}')
                #%%
                model_s = nn.Sequential(feat_extract_dalabelot,  data_class_dalabelot)
                model_s.eval()
                acc_test, bal_acc_s, cm_test = evaluate_clf(model_s, test_loader,n_classes=n_class)
                print(f' {bag_loss_weight}  {bal_acc_s:.4f}')
            
                #%%



                model.eval()
                val_err = error_on_bag_prop(model,val_bags,n_class=n_class)
                if val_err < val_max:
                    val_max = val_err
                    best_bag_loss_weight = bag_loss_weight
                    acc_test, bal_acc_test = evaluate_data_classifier(dalabelot, test_loader, is_target=True, is_ft=False)
            
                print(f' {bag_loss_weight}  {val_max:.4f} {bal_acc_test:.4f}')

            #%%
            print("Accuracy on the test set: ", bal_acc_test)



        list_acc_test.append(acc_test)
        list_bal_acc_test.append(bal_acc_test)
    

        results = {'list_acc_test': list_acc_test,
                    'list_bal_acc_test': list_bal_acc_test, 
                    'args': args,
                    'config': cfg,
                    }
        np.savez(os.path.join(savedir,filesave),**results)

        m_list_bal_acc_test = np.array(list_bal_acc_test).mean()

        print(f'{args.algo} {iter} Mean Accuracy on the test set:  {m_list_bal_acc_test:.4f}')








# %%
