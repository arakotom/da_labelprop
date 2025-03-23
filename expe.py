
#%%
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import warnings
from bagCSI import bagCSI_train
from data import get_toy, get_visda, get_officehome, get_office31
from data import get_mnist_usps, get_usps_mnist
from data import get_bag_stats
from utils import create_data_loader, evaluate_clf, extract_data_label, error_on_bag_prop
from utils_local import evaluate_data_classifier
import yaml 
import sys
from models import FeatureExtractor, DataClassifier, FeatureExtractorDigits, DataClassifierDigits
warnings.filterwarnings("ignore", category=UserWarning) 

def get_model(data, cfg,n_class):
    if data == 'toy' or data == 'visda' or data == 'officehome' or data == 'office31': 
        n_hidden = cfg['model']['n_hidden']
        input_size = cfg['data']['dim']
        feat_extract = FeatureExtractor(input_dim=input_size, n_hidden=n_hidden, output_dim=n_hidden)
        classifier = DataClassifier(input_dim=n_hidden, n_class=n_class)
    elif data == 'mnist_usps' or data == 'usps_mnist':
        feat_extract = FeatureExtractorDigits(channel=1, kernel_size=3, output_dim=128)
        classifier = DataClassifierDigits(input_size=1152, n_class=10)
    return feat_extract, classifier


if __name__ == '__main__':


    
    #sys.argv = ['']
    args = argparse.Namespace()

    parser = argparse.ArgumentParser(description='training llp models')

    # general parameters
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--expe_name', type=str,default="")
    parser.add_argument('--data', type=str, default='office31')
    parser.add_argument('--algo', type=str, default='daLabelWD')
    parser.add_argument('--source_target', type=int, default=4)
    parser.add_argument('--bag_size', type=int, default=50)
    parser.add_argument('--nb_iter', type=int, default=10)
    parser.add_argument('--i_param', type=int, default=0)
    parser.add_argument('--dep_sample', type=int, default=1)
    parser.add_argument('--method', type=str, default='learned')

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
    dep_sample = args.dep_sample
    bag_size = cfg['data']['bag_size']

    print(args.expe_name, data)

    list_acc_test = []
    list_bal_acc_test = []
    

    for iter in range(args.nb_iter):
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
        if data == 'mnist_usps':
            bag_size = cfg['data']['bag_size']
            nb_class_in_bag = cfg['data']['nb_class_in_bag']
            n_class = cfg['data']['n_class']

            source_loader, target_bags  = get_mnist_usps(batch_size=128, drop_last=True,
                        nb_class_in_bag = nb_class_in_bag,
                        bag_size = bag_size,
                        nb_missing_feat = None,
                        dep_sample=dep_sample,
                        apply_miss_feature_source=False)
            if args.expe_name == "":
                savedir = 'results/mnist_usps'
            else:
                savedir = 'results/mnist_usps-' + args.expe_name
        if data == 'usps_mnist':
            bag_size = cfg['data']['bag_size']
            nb_class_in_bag = cfg['data']['nb_class_in_bag']
            n_class = cfg['data']['n_class']

            source_loader, target_bags  = get_usps_mnist(batch_size=128, drop_last=True,
                        nb_class_in_bag = nb_class_in_bag,
                        bag_size = bag_size,
                        nb_missing_feat = None,
                        dep_sample=dep_sample,
                        apply_miss_feature_source=False)
            if args.expe_name == "":
                savedir = 'results/usps_mnist'
            else:
                savedir = 'results/usps_mnist-' + args.expe_name


        if data == 'visda':

            bag_size = cfg['data']['bag_size']
            dim = cfg['data']['dim']
            dim_latent = cfg['model']['dim_latent']
            n_hidden = cfg['model']['n_hidden']
            dist_loss_weight = cfg['daLabelWD']['dist_loss_weight'][args.i_param]
            if args.source_target == 0:
                classe_vec = [0,1,2,3,4,5,6,7,8,9,10,11]
            elif args.source_target == 1:
                classe_vec = [0,4,11]
            n_class = len(classe_vec)
            nb_class_in_bag = n_class
            use_div = False
            source_loader, target_bags  = get_visda(batch_size=128, drop_last=True,
                        nb_class_in_bag = 10,
                        classe_vec=classe_vec,
                        dep_sample=dep_sample,
                        bag_size = bag_size,
                        nb_missing_feat = None,
                        apply_miss_feature_source=False)
            if args.expe_name == "":
                savedir = 'results/visda'
            else:
                savedir = 'results/visda-' + args.expe_name
        if data == 'officehome':
            onfig_file = './configs/officehome.yaml'
            with open(config_file) as file:
                cfg = yaml.load(file, Loader=yaml.FullLoader)
            source = cfg['data']['files'][args.source_target][0]
            target = cfg['data']['files'][args.source_target][1]
            bag_size = cfg['data']['bag_size']
            nb_class_in_bag = cfg['data']['nb_class_in_bag']
            n_class = cfg['data']['n_class']
            dim = cfg['data']['dim']
            dim_latent = cfg['model']['dim_latent']
            n_hidden = cfg['model']['n_hidden']

            source_loader, target_bags = get_officehome(source = source, target = target, batch_size=64,
                                                         drop_last=True,
                         nb_missing_feat = None,
                         dep_sample=dep_sample,
                        nb_class_in_bag = nb_class_in_bag,
                        bag_size = bag_size )

            if args.expe_name == "":
                savedir = 'results/officehome'
            else:
                savedir = 'results/officehome-' + args.expe_name

        if data == 'office31':
            source = cfg['data']['files'][args.source_target][0]
            target = cfg['data']['files'][args.source_target][1]
            bag_size = cfg['data']['bag_size']
            nb_class_in_bag = cfg['data']['nb_class_in_bag']
            n_class = cfg['data']['n_class']
            dim = cfg['data']['dim']
            dim_latent = cfg['model']['dim_latent']
            n_hidden = cfg['model']['n_hidden']

            source_loader, target_bags = get_office31(source = source, target = target, batch_size=64, drop_last=True,
                        nb_missing_feat = None,
                        nb_class_in_bag = nb_class_in_bag,
                        dep_sample=dep_sample,
                        bag_size = bag_size )

            if args.expe_name == "":
                savedir = 'results/office31'
            else:
                savedir = 'results/office31-' + args.expe_name

        print(get_bag_stats(target_bags))

        # --------------------------------------------------------------
        # split the target bags into validation, test and target bags
        # --------------------------------------------------------------
        if len(target_bags) > 10:
            nb_val = len(target_bags)//10
            nb_test = 2*len(target_bags)//10
            val_bags = target_bags[:nb_val]
            test_bags = target_bags[nb_val:nb_test]
            target_bags = target_bags[nb_test:]
        else:
            val_bags = target_bags[1:2]
            test_bags = target_bags[2:4]
            target_bags = target_bags[4:]
    
    


        X_test, y_test = extract_data_label(test_bags, type_data='data', type_label='label')
        test_loader = create_data_loader(X_test, y_test, batch_size=128, shuffle=False,drop_last=False)

        # --------------------------------------------------------------
        # filename for saving the results
        # --------------------------------------------------------------

        filesave = f"data-{data}-algo-{algo}-st-{args.source_target}-dep_sample-{dep_sample}-nb_class_in_bag-{nb_class_in_bag}-bag_size-{bag_size}"

        if args.algo == 'bagLME':
            if args.method == 'learned':
                topk = cfg['bagLME']['topk_lme']
            else:
                topk = cfg['bagLME']['topk_fix']
            filesave += f"-method-{args.method}"
            filesave += f"-topk-{topk}"
            filesave += f"-sw-{cfg['bagLME']['source_weight']:2.3f}"
            filesave += f"-ew-{cfg['bagLME']['ent_weight']:2.3f}"
            
        if args.algo == 'daLabelWD':
            filesave += f"-iter_domain_classifier-{cfg['daLabelWD']['iter_domain_classifier']}"
            filesave += f"-start_align-{cfg['daLabelWD']['start_align']}"
            filesave += f"-lr-{cfg['daLabelWD']['lr']:2.4f}"
        if args.data == 'toy':
            filesave += f"-dim-{dim}-variance-{variance}"
        filesave += f'-seed-{seed}'
        os.makedirs(savedir, exist_ok=True)


        print(filesave)

        acc_test = 0
        bal_acc_test = 0
       #%%
        #----------------------------------------------------------------------------------------
        #       baseline 
        #----------------------------------------------------------------------------------------

        if args.algo == 'bagBase':
            #%%
            print('bagBase')

            learning_rate = cfg['bagCSI']['lr']
            num_epochs = cfg['bagCSI']['n_epochs']
            val_max = n_class
            param_da = 0
            for param_bag in [0.5,1,2]:

                feat_extract, classifier = get_model(data, cfg,n_class)
                model = nn.Sequential(feat_extract,classifier)

                bagCSI_train(feat_extract,classifier, source_loader, target_bags, n_classes=n_class, num_epochs=num_epochs,device=device,
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

                print(f'base {param_bag}  {val_max:.4f} {bal_acc_test:.4f} {val_err:.4f}')


        #%%
        #----------------------------------------------------------------------------------------
        #       bagCSI
        #----------------------------------------------------------------------------------------

        if args.algo == 'bagCSI':
            #%%
            print('bagCSI')

            learning_rate = cfg['bagCSI']['lr']
            num_epochs = cfg['bagCSI']['n_epochs']
            val_max = n_class
            for param_da in [0.1,0.5,1,2]:
                for param_bag in [0.5,1,2]:

                    feat_extract, classifier = get_model(data, cfg,n_class)
                    model = nn.Sequential(feat_extract,classifier)

                    bagCSI_train(feat_extract,classifier, source_loader, target_bags, n_classes=n_class, num_epochs=num_epochs,device=device,
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

                    print(f'bag {param_bag} da {param_da} {val_max:.4f} {bal_acc_test:.4f} {val_err:.4f}')
        #%%

        if algo == 'bagLME':
            #%%
            from models import ResidualPhi, DomainClassifier, DataClassifier, FeatureExtractor
            print('bagLME')
            from bagLME import bagLME_train
            if 1:
                cuda = True if torch.cuda.is_available() else False
                with open(config_file) as file:
                    cfg = yaml.load(file, Loader=yaml.FullLoader)

            x_test, y_test = extract_data_label(target_bags, type_data='data', type_label='label')
            test_loader = create_data_loader(x_test, y_test, batch_size=128, shuffle=False,drop_last=False)

            num_epochs = cfg['bagLME']['n_epochs']
            lr = cfg['bagLME']['lr']
            if args.method == 'learned':
                topk = cfg['bagLME']['topk_lme']
            else:
                topk = cfg['bagLME']['topk_fix']
            source_weight = cfg['bagLME']['source_weight']
            ent_weight = cfg['bagLME']['ent_weight']
            lmesource_weight = cfg['bagLME']['lmesource_weight']       
            val_max = n_class
            for mean_weight in  [0.1,0.5,1,2]:
                for param_bag in [0.5,1,2]:
                    feat_extract, classifier = get_model(data, cfg,n_class)
                    model = nn.Sequential(feat_extract,classifier)
                    bagLME_train(feat_extract,classifier, source_loader, target_bags, n_class=n_class, num_epochs=num_epochs,device=device,
                                source_weight=source_weight,verbose=True, ent_weight=ent_weight,
                                mean_weight=mean_weight,
                                bag_weight=param_bag,
                                method=args.method,
                                lmesource_weight=lmesource_weight,
                                topk=topk,
                                lr=lr)


                    val_err = error_on_bag_prop(model,val_bags,n_class=n_class)
                    if val_err < val_max:
                        val_max = val_err
                        best_param_sw = mean_weight
                        acc_test, bal_acc_test, cm_test = evaluate_clf(model, test_loader,n_classes=n_class)

                    print(f'LME {best_param_sw}  {val_max:.4f} {bal_acc_test:.4f} {val_err:.4f}')





            #%%
        if args.algo == 'daLabelOTPhi':
            #%%
            print('DALabelOTPhi')
            from daLabelOT import daLabelOT
            from models import ResidualPhi, DomainClassifier, DataClassifier, FeatureExtractor
            from utils_local import set_optimizer_data_classifier, set_optimizer_data_classifier_t, set_optimizer_domain_classifier, set_optimizer_feat_extractor, set_optimizer_phi
            from utils_local import estimate_source_proportion

            if 0:
                cuda = True if torch.cuda.is_available() else False
                with open(config_file) as file:
                    cfg = yaml.load(file, Loader=yaml.FullLoader)

            ent_weight = cfg['daLabelOT']['ent_weight']
            clf_t_weight = cfg['daLabelOT']['clf_t_weight']
            div_weight = cfg['daLabelOT']['div_weight']
            n_epochs =  cfg['daLabelOT']['n_epochs']            # total number of epochs
            epoch_start_g = cfg['daLabelOT']['epoch_start_g']   #args.epoch_start_g  # epoch to start retrain the feature extractor
            epoch_train_phi = cfg['daLabelOT']['epoch_train_phi']
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
                domain_class_dalabelot = DomainClassifier(input_dim= dim_latent,n_hidden=n_hidden)
                phi_dalabelot = ResidualPhi(nblocks=nblocks, dim= dim_latent, nl_layer='relu', norm_layer='batch1d', n_branches=1)
                dalabelot = daLabelOT(feat_extract_dalabelot, data_class_dalabelot, phi_dalabelot, source_loader, target_bags,
                                    domain_classifier=domain_class_dalabelot,
                                    cuda=cuda,
                                n_class=n_class, 
                                epoch_start_align=start_align, init_lr=lr,
                                use_div=use_div, n_epochs=n_epochs,
                                clf_t_weight=clf_t_weight, iter=it, epoch_start_g=epoch_start_g,
                                div_weight=div_weight,
                                data_class_t=data_class_t_dalabelot, ent_weight=ent_weight,proportion_S=proportion_S,
                                bag_loss_weight=bag_loss_weight, epoch_train_phi=epoch_train_phi,
                                do_adv=False)
                set_optimizer_phi(dalabelot, optim.Adam(dalabelot.phi.parameters(), lr=0.001, betas=(0.5, 0.999)))
                set_optimizer_data_classifier_t(dalabelot, optim.Adam(dalabelot.data_classifier_t.parameters(), lr=lr, betas=(0.5, 0.999)))
                set_optimizer_feat_extractor(dalabelot, optim.Adam(dalabelot.feat_extractor.parameters(), lr=0, betas=(0.5, 0.999)))
                set_optimizer_data_classifier(dalabelot, optim.Adam(dalabelot.data_classifier.parameters(), lr=lr*10, betas=(0.5, 0.999)))
                set_optimizer_domain_classifier(dalabelot, optim.Adam(dalabelot.domain_classifier.parameters(), lr=lr, betas=(0.5, 0.999)))
                dalabelot.fit()
                #acc_test, bal_acc_test = evaluate_data_classifier(dalabelot, test_loader, is_target=True, is_ft=True)
                #print(f' {bag_loss_weight}  {bal_acc_test:.4f}')
            
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

            if 1:
                cuda = True if torch.cuda.is_available() else False
                with open(config_file) as file:
                    cfg = yaml.load(file, Loader=yaml.FullLoader)


            n_epochs =  cfg['daLabelWD']['n_epochs']            # total number of epochs
            lr = cfg['daLabelWD']['lr']
            start_align = cfg['daLabelWD']['start_align']
            proportion_S = estimate_source_proportion(source_loader, n_clusters=n_class)
            dist_loss_weight = cfg['daLabelWD']['dist_loss_weight'][args.i_param]
            val_max = n_class
            it = iter
            for dist_loss_weight in cfg['daLabelWD']['dist_loss_weight']:
                for bag_loss_weight in cfg['daLabelWD']['bag_loss_weight']:
                    feat_extract_dalabelot = FeatureExtractor(dim, n_hidden=n_hidden, output_dim=dim_latent)
                    data_class_dalabelot = DataClassifier(input_dim= dim_latent, n_class=n_class)
                    domain_class_dalabelot = DomainClassifier(input_dim= dim_latent,n_hidden=n_hidden)

                    
                    dalabelot = daLabelWD(feat_extract_dalabelot, data_class_dalabelot, domain_class_dalabelot, source_loader,
                                        target_bags,
                                    n_class=n_class, 
                                    epoch_start_align=start_align, 
                                    init_lr=lr,
                                    n_epochs=n_epochs,
                                    iter=it, 
                                    iter_domain_classifier=cfg['daLabelWD']['iter_domain_classifier'],
                                    proportion_S=proportion_S,
                                    bag_loss_weight=bag_loss_weight,
                                    dist_loss_weight=dist_loss_weight,)
                    set_optimizer_feat_extractor(dalabelot, optim.Adam(dalabelot.feat_extractor.parameters(), lr=lr, betas=(0.5, 0.999)))
                    set_optimizer_data_classifier(dalabelot, optim.Adam(dalabelot.data_classifier.parameters(), lr=lr, betas=(0.5, 0.999)))
                    set_optimizer_domain_classifier(dalabelot, optim.Adam(dalabelot.domain_classifier.parameters(), lr=lr, betas=(0.5, 0.999)))
                    dalabelot.fit()

                    # #%%
                    model_s = nn.Sequential(feat_extract_dalabelot,  data_class_dalabelot)
                    model_s.eval()


                    model_s = nn.Sequential(feat_extract_dalabelot,  data_class_dalabelot)
                    val_err = error_on_bag_prop(model_s,val_bags,n_class=n_class)
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
