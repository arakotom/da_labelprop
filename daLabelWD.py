#%%
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils_local import evaluate_data_classifier, loop_iterable, set_requires_grad, gradient_penalty, format_list, \
    compute_diff_label, set_lr
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def entropy_loss(v):
    """
    Entropy loss for probabilistic prediction vectors
    """
    return torch.mean(torch.sum(- torch.softmax(v, dim=1) * torch.log_softmax(v, dim=1), 1))

def normalize_gradient(model):
    for param in model.parameters():
        if param.grad is not None:
            param.grad.data = param.grad.data / (param.grad.data.norm() + 1e-8)

class daLabelWD(object):
    def __init__(self, feat_extractor, data_classifier, domain_classifier, source_data_loader, target_bags,
                 grad_scale=1.0, cuda=False,  n_class=10, ts=1, 
                 epoch_start_align=11, n_epochs=100, gamma=10, init_lr=0.001, iter=0,
                 iter_domain_classifier=10, factor_f=1, lr_g_weight=1.0, lr_f_weight=1.0, lr_d_weight=1.0, factor_g=1.0,
                 eval_data_loader=None, proportion_T_gt=None, setting=10, beta_ratio=-1, proportion_S=None,
                 bag_loss_weight=1,
                 dist_loss_weight=1):
        self.feat_extractor = feat_extractor
        self.data_classifier = data_classifier
        #self.setting = setting
        self.iter = iter
        #self.proportion_T_gt = proportion_T_gt
        self.domain_classifier = domain_classifier
        self.source_data_loader = source_data_loader
        self.target_bags = target_bags
        self.eval_data_loader = eval_data_loader
        self.n_class = n_class
        self.lr_g_weight = lr_g_weight
        self.lr_f_weight = lr_f_weight
        self.lr_d_weight = lr_d_weight
        self.proportion_S = proportion_S
        self.cuda = True if torch.cuda.is_available() else False
        self.n_epochs = n_epochs
        self.ent_weight = 0.
        self.criterion = nn.CrossEntropyLoss()
        self.epoch_start_align = epoch_start_align
        #self.epoch_start_g = epoch_start_g
        self.iter_domain_classifier = iter_domain_classifier
        self.gamma = gamma
        #self.grad_scale = grad_scale
        #self.proportion_method = "confusion"
        self.bag_loss_weight = bag_loss_weight
        self.dist_loss_weight = dist_loss_weight
        #self.logger = logger
        #self.cluster_step = compute_cluster_every
        self.init_lr = init_lr
        self.ts = ts
        #self.dataset = dataset
        #self.cluster_param = cluster_param
        #self.prop_factor = 0.5
        #self.factor_f = factor_f
        #self.factor_g = factor_g
        self.optimizer_feat_extractor = optim.SGD(self.feat_extractor.parameters(), lr=0.001)
        self.optimizer_data_classifier = optim.SGD(self.data_classifier.parameters(), lr=0.001)
        self.optimizer_domain_classifier = optim.SGD(self.domain_classifier.parameters(), lr=0.01)

    def fit(self):
        if self.cuda:
            self.feat_extractor.cuda()
            self.data_classifier.cuda()
            self.domain_classifier.cuda()
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        k_critic = self.iter_domain_classifier
        k_prop = 1
        gamma = self.gamma
        self.print_start = True
        self.print_start_g = True

        # proportion_T = torch.ones(self.n_class) / self.n_class
        # if self.beta_ratio == -1:
        #     self.hist_proportion = proportion_T.numpy()

        # Train latent space
        #self.logger.info("--Initialize f, g--")
        print("Initialize f, g")
        for epoch in range(self.n_epochs):
            self.align = (epoch >= self.epoch_start_align)


            S_batches = loop_iterable(self.source_data_loader)
            batch_iterator = zip(S_batches, loop_iterable(self.target_bags))
            batch_iterator_w = zip(S_batches, loop_iterable(self.target_bags))
            iterations = len(self.source_data_loader)


            if self.align:
                if self.print_start:
                    #self.logger.info("--Start Alignment--")
                    print("--Start Alignment--")
                    self.print_start = False

            clf_loss_tot_s, loss_tot, wass_loss_tot, bag_loss_tot = 0, 0, 0, 0
            self.feat_extractor.train()
            self.data_classifier.train()
            if self.align:
                self.domain_classifier.train()



            for batch_idx in range(iterations):
                #(x_s, y_s), (x_t, y_t) = next(batch_iterator)
                #x_s, x_t, y_s, y_t = x_s.to(self.device), x_t.to(self.device), y_s.to(self.device), y_t.to(self.device)

                (x_s, y_s), (bag_t) = next(batch_iterator)

                i_bag = np.random.randint(0, len(self.target_bags))
                bag_t = self.target_bags[i_bag]
                x_t = bag_t['data']
                y_t = bag_t['label']
                proportion_T = bag_t['prop']
                x_s, x_t, y_s, y_t = x_s.to(self.device), x_t.to(self.device), y_s.to(self.device), y_t.to(self.device)




                dist_loss, bag_loss, clf_s_loss = torch.zeros(1).to(self.device), torch.zeros(1).to(self.device), torch.zeros(1).to(self.device)
                if self.align:
                    # Set lr
                    # p = (batch_idx + (epoch - self.epoch_start_align) * len(self.source_data_loader)) / (
                    #         len(self.source_data_loader) * (self.n_epochs - self.epoch_start_align))
                    # lr = float(self.init_lr / (1. + 10 * p) ** 0.75)
                    # set_lr(self.optimizer_domain_classifier, lr * self.lr_d_weight)
                    # set_lr(self.optimizer_data_classifier, lr * self.lr_f_weight)
                    # set_lr(self.optimizer_feat_extractor, lr * self.lr_g_weight)

                    #if self.beta_ratio == -1:
                    source_weight_un = torch.zeros((y_s.size(0), 1)).to(self.device)
                    Ns_class = torch.zeros((self.n_class, 1)).to(self.device)
                    Nt_class = torch.zeros((self.n_class, 1)).to(self.device)
                    for j in range(self.n_class):
                        nb_sample = y_s.eq(j).nonzero().size(0)
                        source_weight_un[y_s == j] = proportion_T[j] / nb_sample   if nb_sample != 0 else 0
                        Ns_class[j] = nb_sample
                        Nt_class[j] = y_t.eq(j).nonzero().size(0)

                    #######################
                    # Train discriminator #
                    #######################
                    wass_loss_tot = 0
                    if self.dist_loss_weight > 0:
                        set_requires_grad(self.feat_extractor, requires_grad=False)
                        set_requires_grad(self.domain_classifier, requires_grad=True)
                        for kk in range(k_critic):
                            (x_s_w, y_s_w), (bag_t_w) = next(batch_iterator_w)
                            i_bag = np.random.randint(0, len(self.target_bags))
                            bag_t_w = self.target_bags[i_bag]


                            x_t_w = bag_t_w['data']
                            x_s_w, x_t_w, y_s_w = x_s_w.to(self.device), x_t_w.to(self.device), y_s_w.to(self.device)

                            source_weight_un_w = torch.zeros((y_s_w.size(0), 1)).to(self.device)
                            for j in range(self.n_class):
                                nb_sample = y_s_w.eq(j).nonzero().size(0)
                                if nb_sample != 0:
                                    source_weight_un_w[y_s_w == j] = proportion_T[j]/ nb_sample
                            with torch.no_grad():
                                z_w = self.feat_extractor(torch.cat((x_s_w, x_t_w), 0))
                                s_w = z_w[:x_s_w.shape[0]]
                                t_w = z_w[x_s_w.shape[0]:]
                                if s_w.size(0) >= t_w.size(0):
                                    ind = torch.randperm(s_w.size(0))[:t_w.size(0)]
                                    s_w = s_w[ind]
                                    source_weight_un_w = source_weight_un_w[ind]
                                elif s_w.size(0) < t_w.size(0):
                                    ind = torch.randperm(t_w.size(0))[:s_w.size(0)]
                                    t_w = t_w[ind]
                            #print(s_w.shape, t_w.shape)
                            gp = gradient_penalty(self.domain_classifier, s_w, t_w, self.cuda)
                            critic_w = self.domain_classifier(torch.cat((s_w, t_w), 0))
                            critic_s_w, critic_t_w = critic_w[:s_w.shape[0]], critic_w[s_w.shape[0]:]
                            #print(critic_s_w.shape, critic_t_w.shape, source_weight_un_w.shape)
                            wasserstein_distance_w = (critic_s_w * source_weight_un_w.detach()).sum() - critic_t_w.mean()

                            critic_cost = - wasserstein_distance_w + gamma * gp
                            self.optimizer_domain_classifier.zero_grad()
                            critic_cost.backward()
                            self.optimizer_domain_classifier.step()
                            wass_loss_tot += wasserstein_distance_w.item()

                    ##############
                    # Train f, g #
                    ##############
                    set_requires_grad(self.data_classifier, requires_grad=True)
                    set_requires_grad(self.feat_extractor, requires_grad=True)
                    set_requires_grad(self.domain_classifier, requires_grad=False)
                    z = self.feat_extractor(torch.cat((x_s, x_t), 0))

                    # Classif
                    #self.criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(torch.Tensor(proportion_T) / self.proportion_S).to(self.device))
                    # there is no label shift in  the target data, just the bag that have different proportion
                    self.criterion = nn.CrossEntropyLoss()
                    clf_s_loss = self.criterion(self.data_classifier(z[:x_s.shape[0]]), y_s)

                    # Alignment
                    critic = self.domain_classifier(z)
                    critic_s, critic_t = critic[:x_s.shape[0]], critic[x_s.shape[0]:]
                    dist_loss = ((critic_s * source_weight_un.detach()).sum() - critic_t.mean())

                    # Bag loss
                    outputs_target = self.data_classifier(z[x_s.shape[0]:])
                    outputs_target = torch.softmax(outputs_target, dim=1)
                    bag_loss = torch.mean(torch.abs(outputs_target.mean(dim=0) - torch.Tensor(proportion_T).to(self.device)))
                        
                    #
                    # output_class_t = self.data_classifier(z[x_s.shape[0]:])
                    # ent_loss = self.ent_weight * entropy_loss(output_class_t)


                    loss = clf_s_loss + self.dist_loss_weight*dist_loss + bag_loss*self.bag_loss_weight
                    #loss += ent_loss
                    self.optimizer_data_classifier.zero_grad()
                    self.optimizer_feat_extractor.zero_grad()
                    loss.backward() 
                    self.optimizer_data_classifier.step()
                    self.optimizer_feat_extractor.step()

                else:
                    set_requires_grad(self.data_classifier, requires_grad=True)
                    set_requires_grad(self.feat_extractor, requires_grad=True)
                    z = self.feat_extractor(torch.cat((x_s, x_t), 0))
                    self.criterion = nn.CrossEntropyLoss()
                    clf_s_loss = self.criterion(self.data_classifier(z[:x_s.shape[0]]), y_s)

                    # Bag loss
                    outputs_target = self.data_classifier(z[x_s.shape[0]:])
                    outputs_target = torch.softmax(outputs_target, dim=1)
                    bag_loss = torch.mean(torch.abs(outputs_target.mean(dim=0) - torch.Tensor(proportion_T).to(self.device)))
                        

                    loss = clf_s_loss + bag_loss*self.bag_loss_weight

                    self.optimizer_feat_extractor.zero_grad()
                    self.optimizer_data_classifier.zero_grad()
                    loss.backward()
                    self.optimizer_feat_extractor.step()
                    self.optimizer_data_classifier.step()

                loss_tot += loss.item()
                clf_loss_tot_s += clf_s_loss.item()
                bag_loss_tot += bag_loss.item()

            loss_tot /= iterations
            clf_loss_tot_s /= iterations
            wass_loss_tot /= iterations
            bag_loss_tot /= iterations
            #self.logger.info('{} {} {} s{} Iter {} Epoch {}/{} \tTotal: {:.6f} L_S: {:.6f} DistL:{:.6f} WassL:{:.6f}'.format(self.ts,
            #    comment, self.dataset, self.setting, self.iter, epoch, self.n_epochs, loss_tot, clf_loss_tot_s, dist_loss_tot, wass_loss_tot))
            print('Iter {} Epoch {}/{} \tTotal: {:.6f} L_S: {:.6f} BagL:{:.6f} WassL:{:.6f}'.format(self.iter, epoch, self.n_epochs, loss_tot, clf_loss_tot_s, bag_loss_tot, wass_loss_tot))
            if (epoch + 1) % 5 == 0:
                evaluate_data_classifier(self, self.source_data_loader, is_target=False, verbose=False)
                #evaluate_data_classifier(self, self.eval_data_loader, is_target=True)
                #if self.beta_ratio == -1:
                #    self.logger.info(f"pT(Y) {self.proportion_method}: {format_list(proportion_T, 4)}")

if __name__ == '__main__':
    
    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if 0:
        from data import get_toy
        import matplotlib.pyplot as plt
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
        lr = cfg['bagLME']['lr']
        num_epochs = cfg['bagLME']['n_epochs']
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
        dim = cfg['data']['dim']
        dim_latent = cfg['model']['dim_latent']
        n_hidden = cfg['model']['n_hidden']
        lr = cfg['bagLME']['lr']
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
    elif 0: 
        config_file = './configs/mnist_usps.yaml'
        import yaml
        from data import get_mnist_usps
        cuda = True if torch.cuda.is_available() else False
        with open(config_file) as file:
            cfg = yaml.load(file, Loader=yaml.FullLoader)
        bag_size = cfg['data']['bag_size']
        dim_latent = cfg['model']['dim_latent']
        n_hidden = cfg['model']['n_hidden']
        lr = cfg['bagLME']['lr']
        num_epochs = 30
        n_class = 10
        use_div = False
        source_loader, target_bags  = get_mnist_usps(batch_size=256, drop_last=True,
                    nb_class_in_bag = 10,
                    bag_size = 50,
                    nb_missing_feat = None,
                    apply_miss_feature_source=False)

    x_train, y_train = source_loader.dataset.tensors
    large_source_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=1024, shuffle=True, drop_last=False)
    large_source_loader = loop_iterable(large_source_loader)
    input_size = x_train.size(1)
    # Initialize the model, loss function, and optimizer
    dim = input_size

    
    from models import DomainClassifier, DataClassifier, FeatureExtractor
    from utils_local import set_optimizer_data_classifier, set_optimizer_domain_classifier, set_optimizer_feat_extractor
    from utils_local import estimate_source_proportion

    feat_extract_dalabelot = FeatureExtractor(dim, n_hidden=n_hidden, output_dim=dim_latent)
    data_class_dalabelot = DataClassifier(input_dim= dim_latent, n_class=n_class)
    domain_class_dalabelot = DomainClassifier(input_dim= dim_latent,n_hidden=n_hidden)
    cuda = True if torch.cuda.is_available() else False

    lr_f, lr_g, lr_d = 0.001, 0.001, 0.001
    ent_weight = clf_t_weight = div_weight = 0.
    bag_loss_weight = 1
    dist_loss_weight = 0.001
    it = 0
    use_div = False #(dataset == "office" or dataset == "visda")
    opt={'start_align': 0,'lr': 0.001}
    proportion_S = estimate_source_proportion(source_loader, n_clusters=n_class)

    
    dalabelot = daLabelWD(feat_extract_dalabelot, data_class_dalabelot, domain_class_dalabelot, source_loader, target_bags,
                           cuda=cuda,
                     n_class=n_class, 
                    epoch_start_align=opt["start_align"], init_lr=opt["lr"],
                    n_epochs=num_epochs,
                    iter=it, 
                    iter_domain_classifier=2,
                    proportion_S=proportion_S,
                    dist_loss_weight=dist_loss_weight,
                    bag_loss_weight=bag_loss_weight,)
    set_optimizer_feat_extractor(dalabelot, optim.Adam(dalabelot.feat_extractor.parameters(), lr=lr_g, betas=(0.5, 0.999)))
    set_optimizer_data_classifier(dalabelot, optim.Adam(dalabelot.data_classifier.parameters(), lr=lr_f, betas=(0.5, 0.999)))
    set_optimizer_domain_classifier(dalabelot, optim.Adam(dalabelot.domain_classifier.parameters(), lr=lr_d, betas=(0.5, 0.999)))

    dalabelot.fit()

    




    
    from utils import evaluate_clf, create_data_loader, extract_data_label

    x_test, y_test = extract_data_label(target_bags, type_data='data', type_label='label')
    test_loader = create_data_loader(x_test, y_test, batch_size=128, shuffle=False,drop_last=False)

    acc, bal_acc_1, cm = evaluate_clf(nn.Sequential(feat_extract_dalabelot,data_class_dalabelot), test_loader,n_classes=n_class,return_pred=False)
    #acc, bal_acc_2, cm = evaluate_clf(nn.Sequential(feat_extract,classifier_2), test_loader,n_classes=n_class,return_pred=False)
    print(f'Balanced Accuracy S+T: {bal_acc_1:.4f}')


# %%
