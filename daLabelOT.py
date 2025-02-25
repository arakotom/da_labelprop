#%%

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils_local import evaluate_data_classifier, loop_iterable, set_requires_grad, gradient_penalty, format_list, \
                     set_lr, entropy_loss
import ot
#from proportion_estimators import estimate_proportion
def dist_torch(x1,x2):
    x1p = x1.pow(2).sum(1).unsqueeze(1)
    x2p = x2.pow(2).sum(1).unsqueeze(1)
    prod_x1x2 = torch.mm(x1,x2.t())
    distance = x1p.expand_as(prod_x1x2) + x2p.t().expand_as(prod_x1x2) -2*prod_x1x2
    return distance 


class daLabelOT(object):
    def __init__(self, feat_extractor, data_classifier, phi, source_data_loader, target_bags, 
                 grad_scale=1.0, cuda=False, logger=None, n_class=10, domain_classifier=None, 
                 epoch_start_align=11, 
                 cluster_param="ward", epoch_start_g=30,
                 n_epochs=100, init_lr=0.001, iter_domain_classifier=10, lr_g_weight=1.0,
                 lr_f_weight=1.0, lr_phi_weight=1.0, eval_data_loader=None, iter=0,
                 data_class_t=None, ent_weight=0.1, div_weight=0.1, clf_t_weight=0.3, bag_loss_weight=0.1,
                 use_div=False,proportion_S=None,
):
        self.feat_extractor = feat_extractor
        self.data_classifier = data_classifier
        self.data_classifier_t = data_class_t
        self.phi = phi
        self.iter = iter
        #self.proportion_T_gt = proportion_T_gt
        #self.domain_classifier = domain_classifier
        self.source_data_loader = source_data_loader
        self.target_bags = target_bags
        self.eval_data_loader = eval_data_loader
        self.proportion_S = proportion_S
        self.n_class = n_class
        self.ent_weight = ent_weight
        # adjusting the different learning rates by weighting
        self.lr_g_weight = lr_g_weight
        self.lr_f_weight = lr_f_weight
        self.lr_phi_weight = lr_phi_weight
        #self.ot_weight = ot_weight
        #self.ot_weight_init = ot_weight
        #self.grad_scale = grad_scale
        self.clf_t_weight = clf_t_weight
        self.cuda = cuda
        self.n_epochs = n_epochs
        self.criterion = nn.CrossEntropyLoss()
        self.epoch_start_align = epoch_start_align
        self.epoch_start_g = epoch_start_g
        #self.iter_domain_classifier = iter_domain_classifier
        #self.gamma = gamma
        #self.proportion_method = "confusion"
        self.logger = logger
        #self.cluster_step = cluster_step
        self.init_lr = init_lr
        #self.ts = ts
        #self.dataset = dataset
        #self.cluster_param = cluster_param
        #self.prop_factor = 0.5
        self.optimizer_feat_extractor = optim.SGD(self.feat_extractor.parameters(), lr=init_lr)
        self.optimizer_data_classifier = optim.SGD(self.data_classifier.parameters(), lr=init_lr)
        self.optimizer_data_classifier_t = optim.SGD(self.data_classifier_t.parameters(), lr=init_lr)
        #self.optimizer_domain_classifier = optim.SGD(self.domain_classifier.parameters(), lr=init_lr)
        self.optimizer_phi = optim.SGD(self.phi.parameters(), lr=0.001)
        self.use_div = use_div
        self.div_weight = div_weight
        self.bag_loss_weight = bag_loss_weight  

    def fit(self):
        if self.cuda:
            self.feat_extractor.cuda()
            self.data_classifier.cuda()
            self.data_classifier_t.cuda()
            #self.domain_classifier.cuda()
            self.phi.cuda()
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        #k_critic = self.iter_domain_classifier
        #k_prop = 1
        #gamma = self.gamma
        self.print_start = True
        self.print_start_g = True
        self.use_phi = False

        #proportion_T = torch.ones(self.n_class) / self.n_class
        #self.hist_proportion = proportion_T.numpy()

        # Train latent space
        #self.logger.info("--Initialize f, g--")
        print("Initialize feat, source classifier")
        for epoch in range(self.n_epochs):
            #self.recluster = ((self.epoch_start_g > epoch > self.epoch_start_align and ((epoch - self.epoch_start_align) % 2) == 0) or
            #                  (epoch == self.epoch_start_g) or (epoch >= self.epoch_start_g and (epoch % self.cluster_step) == 0))
            self.align = (epoch >= self.epoch_start_align)

            S_batches = loop_iterable(self.source_data_loader)
            batch_iterator = zip(S_batches, loop_iterable(self.target_bags))
            batch_iterator_w = zip(S_batches, loop_iterable(self.target_bags))
            iterations = len(self.source_data_loader)

            if self.align:
                if self.print_start:
                    #self.logger.info("--Train phi--")
                    print("Train phi")
                    self.print_start = False
                if self.print_start_g and epoch >= self.epoch_start_g:
                    #self.logger.info("--Train g--")
                    self.print_start_g = False
                    print("Train feat extractor")
            #dist_loss_tot, ot_loss_tot, clf_loss_tot_s, clf_loss_tot_t, loss_tot, wass_loss_tot, ent_loss_tot = \
            #    0, 0, 0, 0, 0, 0, 0
            clf_loss_tot_s, clf_loss_tot_t, loss_tot, wass_loss_tot, ent_loss_tot, bag_loss_tot = \
                0, 0, 0, 0, 0, 0

            self.feat_extractor.train()
            if self.align:
                self.phi.train()
                #self.domain_classifier.train()
                self.data_classifier_t.train()
            else:
                self.data_classifier.train()

            if epoch == self.epoch_start_align:
                self.data_classifier_t.load_state_dict(self.data_classifier.state_dict())


            self.use_phi = self.align

            for batch_idx in range(iterations):
                (x_s, y_s), (bag_t) = next(batch_iterator)
                x_t = bag_t['data']
                y_t = bag_t['label']
                proportion_T = bag_t['prop']
                x_s, x_t, y_s, y_t = x_s.to(self.device), x_t.to(self.device), y_s.to(self.device), y_t.to(self.device)

                # ent_loss, dist_loss, ot_loss, clf_s_loss, clf_t_loss, pl_loss, N_tpl = torch.zeros(1).to(self.device),\
                #     torch.zeros(1).to(self.device), torch.zeros(1).to(self.device), torch.zeros(1).to(self.device), \
                #     torch.zeros(1).to(self.device), torch.zeros(1).to(self.device), 0
                ent_loss, clf_s_loss, clf_t_loss, bag_loss = torch.zeros(1).to(self.device),\
                    torch.zeros(1).to(self.device), torch.zeros(1).to(self.device),torch.zeros(1).to(self.device)

                if self.align:
                    # do all the alignment stuff for domain adaptation
                    # Set lr
                    p = (batch_idx + (epoch - self.epoch_start_align) * len(self.source_data_loader)) / (
                            len(self.source_data_loader) * (self.n_epochs - self.epoch_start_align))
                    lr = float(self.init_lr / (1. + 10 * p) ** 0.75)
                    set_lr(self.optimizer_data_classifier_t, lr * self.lr_f_weight)
                    set_lr(self.optimizer_feat_extractor, lr * self.lr_g_weight)
                    set_lr(self.optimizer_phi, lr * self.lr_phi_weight)

                    source_weight_un = torch.zeros((y_s.size(0), 1)).to(self.device)
                    weight_class = torch.zeros((y_s.size(0), 1)).to(self.device)
                    Ns_class = torch.zeros((self.n_class, 1)).to(self.device)
                    Nt_class = torch.zeros((self.n_class, 1)).to(self.device)
                    for j in range(self.n_class):
                        nb_sample = y_s.eq(j).nonzero().size(0)
                        source_weight_un[y_s == j] = proportion_T[j] / nb_sample  if nb_sample != 0 else 0
                        weight_class[y_s == j] = 1 / nb_sample if nb_sample != 0 else 0
                        Ns_class[j] = nb_sample
                        Nt_class[j] = y_t.eq(j).nonzero().size(0)

                    # #############
                    # # Train phi #
                    # #############
                    set_requires_grad(self.phi, requires_grad=True)
                    set_requires_grad(self.feat_extractor, requires_grad=False)
                    k_critic = 50 if epoch == self.epoch_start_align else 3
                    for _ in range(k_critic):
                        (x_s_w, y_s_w), (bag_t_w) = next(batch_iterator_w)
                        x_t_w = bag_t_w['data']
                        bag_prop = bag_t_w['prop']
                        x_s_w, x_t_w, y_s_w = x_s_w.to(self.device), x_t_w.to(self.device), y_s_w.to(self.device)
                        with torch.no_grad():
                            z_nograd = self.feat_extractor(torch.cat((x_s_w, x_t_w), 0))
                            zt_nograd = z_nograd[x_s.shape[0]:]
                            zs_nograd = z_nograd[:x_s.shape[0]]
                        phi_z_s_nograd, rs_nograd = self.phi(zs_nograd.detach())
                        
                        n_1 = x_s_w.size(0)  # Xa is the anchor points  x_s_w
                        n_2 = x_t_w.size(0)     # X is the target points x_t_w
                        a = torch.from_numpy(ot.unif(n_1)).float()
                        b = torch.from_numpy(ot.unif(n_2)).float()
                        for i_c in range(self.n_class):
                            ind = torch.where(y_s_w  == i_c)[0]
                            n_in_class = len(ind)          
                            a[ind] = bag_prop[i_c]/n_in_class if n_in_class > 0 else 0.0
                        b /= torch.sum(b)
                        a /= torch.sum(a)
                        b = b.float()
                        a = a.float()
                        # do PCA of zt_nograd
                        # from sklearn.decomposition import PCA
                        # pca = PCA(n_components=20)
                        # zt_nograd_pca = pca.fit_transform(zt_nograd.detach().cpu().numpy())
                        # zt_nograd_pca = torch.from_numpy(zt_nograd_pca).float().to(self.device)
                        # zs_nograd_pca = pca.fit_transform(zs_nograd.detach().cpu().numpy())
                        # zs_nograd_pca = torch.from_numpy(zs_nograd_pca).float().to(self.device)
                        # M = dist_torch(zs_nograd_pca,zt_nograd_pca)

                        M = dist_torch(zs_nograd,zt_nograd)
                        with torch.no_grad():
                            gamma = ot.emd(a,b,M)
                        # barycentric mapping
                        zs_map = torch.mm(gamma,zt_nograd)*(1/(gamma@torch.ones(n_2))).reshape(-1,1)
                        # keep only samples that have been mapped (with a >0)
                        ind = torch.where(torch.isnan(zs_map)==False)   
                        loss = torch.sum(torch.abs(phi_z_s_nograd[ind] - zs_map[ind])**2)                                            
                        self.optimizer_phi.zero_grad()
                        loss.backward()
                        self.optimizer_phi.step()
                        wass_loss_tot += loss.item()/k_critic
                       
                    ###############
                    # Train feature extractor, classifier in the target domain #
                    ###############
                    self.train_g = False if (self.epoch_start_g > epoch >= self.epoch_start_align) else True
                    set_requires_grad(self.data_classifier, requires_grad=False)
                    set_requires_grad(self.data_classifier_t, requires_grad=True)
                    set_requires_grad(self.feat_extractor, requires_grad=self.train_g)
                    set_requires_grad(self.phi, requires_grad=False)

                    z = self.feat_extractor(torch.cat((x_s, x_t), 0))
                    z_s, z_t = z[:x_s.shape[0]], z[x_s.shape[0]:]
                    phi_z_s = self.phi(z_s)[0]

                    # Classifier loss on target domain (phi) but on source data
                    weight = torch.Tensor([proportion_T[kk] / self.proportion_S[kk] for kk in range(self.n_class)])
                    self.criterion = nn.CrossEntropyLoss(weight=weight)
                    clf_t_loss = self.criterion(self.data_classifier_t(phi_z_s.detach()), y_s)
                    #clf_t_loss *= self.clf_t_weight

                    # bag loss on target domain
                    outputs_target = self.data_classifier_t(z_t)
                    outputs_target = torch.softmax(outputs_target, dim=1)
                    bag_loss = torch.mean(torch.abs(outputs_target.mean(dim=0) - torch.Tensor(bag_prop)))
                    #print(outputs_target.mean(dim=0), torch.Tensor(bag_prop))
                    #bag_loss *= self.bag_loss_weight

                    # Entropy on target
                    output_class_t = self.data_classifier_t(z_t)
                    ent_loss = self.ent_weight * entropy_loss(output_class_t)
                    if self.use_div:
                        msoftmax = nn.Softmax(dim=1)(output_class_t).mean(dim=0)
                        ent_loss -= self.div_weight * torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))

                    if self.train_g:
                        # Source
                        self.criterion = nn.CrossEntropyLoss()
                        source_preds = self.data_classifier(z_s)
                        clf_s_loss = self.criterion(source_preds, y_s)

                    loss = clf_s_loss + clf_t_loss*self.clf_t_weight + ent_loss + bag_loss*self.bag_loss_weight
                    if self.train_g:
                        self.optimizer_feat_extractor.zero_grad()
                        #self.optimizer_data_classifier.zero_grad()
                    self.optimizer_data_classifier_t.zero_grad()
                    loss.backward()
                    if self.train_g:
                        self.optimizer_feat_extractor.step()
                        #self.optimizer_data_classifier.step()
                    self.optimizer_data_classifier_t.step()
                else:
                    set_requires_grad(self.data_classifier, requires_grad=True)
                    set_requires_grad(self.feat_extractor, requires_grad=True)
                    z = self.feat_extractor(torch.cat((x_s, x_t), 0))
                    source_preds = self.data_classifier(z[:x_s.shape[0]])
                    self.criterion = nn.CrossEntropyLoss()
                    clf_s_loss = self.criterion(source_preds, y_s)

                    loss = clf_s_loss

                    self.optimizer_feat_extractor.zero_grad()
                    self.optimizer_data_classifier.zero_grad()
                    loss.backward()
                    self.optimizer_feat_extractor.step()
                    self.optimizer_data_classifier.step()

            loss_tot += loss.item()
            clf_loss_tot_s += clf_s_loss.item()
            clf_loss_tot_t += clf_t_loss.item()
            bag_loss_tot += bag_loss.item()
            ent_loss_tot += ent_loss.item()

            #self.logger.info(
            #    '{} OSTAR {} s{} Iter {} Epoch {}/{} \tTotal: {:.6f} L_S: {:.6f} L_T: {:.6f} DistL:{:.6f} WassL:{:.6f} '
            #    'OTL:{:.6f} H:{:.6f}'.format(self.ts, self.dataset, self.setting, self.iter, epoch, self.n_epochs,
            #    loss_tot, clf_loss_tot_s, clf_loss_tot_t, dist_loss_tot, wass_loss_tot, ot_loss_tot, ent_loss_tot))
            print(
                'Iter {} Epoch {}/{} \tTotal: {:.6f} L_S: {:.6f} L_T: {:.6f} WassL:{:.6f} BagL:{:.6f} H:{:.6f}'.format(self.iter, epoch, self.n_epochs,
                loss_tot, clf_loss_tot_s, clf_loss_tot_t,  wass_loss_tot, bag_loss_tot, ent_loss_tot))
            
            if (epoch + 1) % 5 == 0:
                evaluate_data_classifier(self, self.source_data_loader, is_target=False, verbose=False)
                #evaluate_data_classifier(self, self.eval_data_loader, is_target=True, is_ft=self.align)
                #self.logger.info(f"pT(Y) {self.proportion_method}: {format_list(proportion_T, 4)}")

if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    from data import get_toy

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

    dim = 2

    source_loader, target_bags = get_toy(apply_miss_feature_source=False,
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
    from models import ResidualPhi, DomainClassifier, DataClassifier, FeatureExtractor
    from utils_local import set_optimizer_data_classifier, set_optimizer_data_classifier_t, set_optimizer_domain_classifier, set_optimizer_feat_extractor, set_optimizer_phi
    from utils_local import estimate_source_proportion
    dim_latent = 10
    n_hidden = 100
    n_class = 3
    feat_extract_dalabelot = FeatureExtractor(dim, n_hidden=n_hidden, output_dim=dim_latent)
    data_class_dalabelot = DataClassifier(input_dim= dim_latent, n_class=n_class)
    data_class_t_dalabelot = DataClassifier(input_dim= dim_latent, n_class=n_class)
    phi_dalabelot = ResidualPhi(nblocks=5, dim= dim_latent, nl_layer='relu', norm_layer='batch1d', n_branches=1)
    domain_class_dalabelot = DomainClassifier(input_dim= dim_latent,n_hidden=n_hidden)
    cuda = True if torch.cuda.is_available() else False

    lr_f, lr_g, lr_phi, lr_d = 0.01, 0.01, 0.01, 0.01
    ent_weight = clf_t_weight = div_weight = 0.1
    bag_loss_weight = 0.01
    n_epochs_star = 150
    epoch_start_g = 50
    it = 0
    use_div = False #(dataset == "office" or dataset == "visda")
    opt={'start_align': 5,'lr': 0.001}
    proportion_S = estimate_source_proportion(source_loader, n_clusters=3)

    
    dalabelot = daLabelOT(feat_extract_dalabelot, data_class_dalabelot, phi_dalabelot, source_loader, target_bags,
                           cuda=cuda,
                     n_class=n_class, 
                    epoch_start_align=opt["start_align"], init_lr=opt["lr"],
                    use_div=use_div, n_epochs=n_epochs_star,
                    clf_t_weight=clf_t_weight, iter=it, epoch_start_g=epoch_start_g,
                    div_weight=div_weight,
                    data_class_t=data_class_t_dalabelot, ent_weight=ent_weight,proportion_S=proportion_S,
                    bag_loss_weight=bag_loss_weight)
    set_optimizer_phi(dalabelot, optim.Adam(dalabelot.phi.parameters(), lr=lr_phi, betas=(0.5, 0.999)))
    set_optimizer_data_classifier_t(dalabelot, optim.Adam(dalabelot.data_classifier_t.parameters(), lr=lr_f, betas=(0.5, 0.999)))
    set_optimizer_feat_extractor(dalabelot, optim.Adam(dalabelot.feat_extractor.parameters(), lr=lr_g, betas=(0.5, 0.999)))
    set_optimizer_data_classifier(dalabelot, optim.Adam(dalabelot.data_classifier.parameters(), lr=lr_f, betas=(0.5, 0.999)))
    dalabelot.fit()
# %%

    from utils import  create_data_loader
    x_test, y_test = extract_data_label(target_bags)
    test_loader= create_data_loader(x_test, y_test, batch_size=32,shuffle=False,drop_last=False)
    acc_train, map_train = evaluate_data_classifier(dalabelot, source_loader, is_target=False, is_ft=False)
    acc_test, map_test = evaluate_data_classifier(dalabelot, test_loader, is_target=True, is_ft=True)
    print(f"Train accuracy: {acc_train:.4f} / Test accuracy: {acc_test:.4f}")


# %%
    def extract_feature_extr(net,train_loader,device='cpu'):
        net.eval()
        train_feature = []
        train_label = []
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            label = targets.cpu()
            inputs, targets = inputs.to(device), targets.to(device)        

            feature = net(inputs)
            feature = feature.detach().cpu()
            train_feature.append(feature)
            train_label.append(label)
                            
        train_label = torch.cat(train_label)
        train_feature = torch.cat(train_feature, 0)
        
        net.train()

        return train_feature, train_label

    x_s, y_s = extract_feature_extr(ostar.feat_extractor,source_loader)
    x_t, y_t = extract_feature_extr(ostar.feat_extractor,test_loader)
    x_s_map = ostar.phi(x_s)[0].detach().cpu().numpy()
    #plt.scatter(x_s[:, 0], x_s[:, 1], c=y_s, cmap='viridis')
    plt.scatter(x_t[:, 0], x_t[:, 1], c=y_t, cmap='viridis', marker='x',alpha=0.2)
    plt.scatter(x_s_map[:, 0], x_s_map[:, 1], c=y_s, cmap='viridis', marker='s',alpha=0.5)

