import os
import time
from collections import OrderedDict, defaultdict
import numpy as np
from scipy.stats import entropy as KL

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['agg.path.chunksize'] = 100000
import matplotlib.pyplot as plt 
from matplotlib import cm

import seaborn as sns
import pandas as pd

import torch
from torch import nn, optim
from torch.nn import functional as F

from torchvision import transforms

from utils import display_progress
import warnings
warnings.simplefilter("ignore")


def normalize(influences):
    """Normalize influences to [-1,1]"""

    maximum = influences.max()
    minimum = influences.min()
    assert maximum > 0
    if minimum < -maximum: 
        scale = -minimum
    else:
        scale = maximum
    return influences / scale


def cos(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_auc_score(inds, flip_ind, rescale=False):
    """
    Compute AUC score of outlier detector

    Arguments:
        inds: list, indices of data in outlier score order (e.g. self-inf)
        flip_ind: list, indices of ground-truth outliers
        rescale: bool, whether to rescale to perfect 0-1

    Returns:
        AUC: float between 0 and 1, the larger the better
        rates: list of discover rates at each fraction of training data checked
        # positions: list of ranks of ground-truth outliers
    """
    set_flip_ind = set(flip_ind)
    N = len(inds)  # number of training samples
    Nflip = len(flip_ind)  # number of flipped samples

    rates = [0.0 for _ in range(N)]
    for k in range(N):
        rates[k] = (inds[k] in set_flip_ind) / float(Nflip) + (rates[k-1] if k > 0 else 0)

    if rescale:
        for k in range(Nflip):
            rates[k] *= Nflip / (k + 1.0)

    # positions = [i for i, ind_x in enumerate(inds) if ind_x in set_flip_ind]
    # auc = 1 - sum(positions) / (len(flip_ind) * len(inds))
    
    auc = np.mean(rates)
    return auc, rates


def self_inf_distribution_by_elbo(path, dataset, data_loader, influences, loss_fn):
    """
    Plot distribution of self influences vs elbo 

    Arguments:
        path: str, figure output path
        dataset: str, dataset name
        data_loader: pytorch dataloader
        influences: np array (n_train, )
        loss_fn: loss function
    """
    print("Computing self influence distribution by ELBO")

    influences = normalize(influences)
    influences = influences.reshape((-1,))

    # get losses
    losses = []
    for i, z in enumerate(data_loader.dataset):
        if dataset[:5] in ['mnist', 'cifar']:
            z = z[0]
        with torch.no_grad():
            losses.append(-loss_fn(z))
        display_progress('Computing loss:', i, len(data_loader))
    losses = np.array(losses)

    # plot scatter
    from sklearn.preprocessing import minmax_scale
    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    model.fit(losses.reshape((-1,1)), influences.reshape((-1,1)))
    R2 = model.score(losses.reshape((-1,1)), influences.reshape((-1,1)))
    x_new = np.linspace(min(losses), max(losses), 100)
    y_new = model.predict(x_new[:, np.newaxis])

    fig, ax = plt.subplots(figsize=(4,2))
    cmap = getattr(cm, 'plasma_r', cm.hot_r)
    s = 0.5 if len(data_loader) <= 5000 else 0.1
    ax.scatter(losses, influences, s=s, alpha=0.8, c=cmap(minmax_scale(losses)))
    ax.plot(x_new, y_new, linestyle='--', label=r'linear reg ($R^2=${:.2f})'.format(R2))
    ax.set_xlabel(r'$-\ell_{\beta}(x_i)$')
    ax.set_ylabel(r'VAE-TracIn($x_i$, $x_i$)')
    plt.legend(loc=1)
    plt.locator_params(axis='x', nbins=2)
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'selfinf_byELBO.jpg'), dpi=400)
    
    # plot density
    df = pd.DataFrame({'x': losses, 'y': influences})
    ax = sns.jointplot(x=df.x, y=df.y, cmap="Blues", shade=True, kind='kde')
    ax.set_axis_labels(r'$-\ell_{\beta}(x_i)$', r'TracIn($x_i$, $x_i$)')
    plt.savefig(os.path.join(path, 'selfinf_kde_byELBO.pdf'))


def train_inf_distribution_by_label(path, data_loader, all_labels, test_labels, influences):
    """
    Plot distribution of training influences by labels

    Arguments:
        path: str, figure output path
        data_loader: pytorch dataloader
        all_labels: list, [0..9] for mnist
        test_labels: np array (n_test, )
        influences: np array (n_test, n_train)
    """
    assert len(data_loader) == influences.shape[1]
    assert len(test_labels) == influences.shape[0]
    print("Computing training influence distributions")

    influences = normalize(influences)

    # statistics
    data_labels = [int(x[1]) for x in data_loader.dataset]

    inf_dic = defaultdict(list)
    for j, influence in enumerate(influences):
        for i, score in enumerate(influence):
            label_i = data_labels[i]
            label_j = test_labels[j]
            inf_dic[(label_i, label_j)].append(score)

    # plot
    plt.figure(figsize=(20, 6))
    
    nrow, ncol = 2, (len(all_labels)+1) // 2
    for label in all_labels:
        plt.subplot(nrow, ncol, label+1)
        diff_class_inf = []
        for label_i in all_labels:
            diff_class_inf += inf_dic[(label_i, label)] if label_i != label else []
        plt.hist(diff_class_inf, 150, density=True, alpha=0.6, label=r'$y_i\neq${}'.format(label))
        plt.hist(inf_dic[(label, label)], 150, density=True, alpha=0.6, label=r'$y_i=${}'.format(label), color='red')
        plt.legend()
        plt.yticks([])
        plt.xlabel(r'TracIn($x_i$, $z_j$) (label($z_j$)={})'.format(label))

    plt.subplot_tool()
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'traininf_distribution.pdf'))


def get_inf_vs_latent_dist(feat_net, dataset, path, data_loader, z_tests, influences, gpu):
    """
    Plot distribution of influences vs distance in latent space

    Arguments:
        feat_net: torch neural network to compute latent
        dataset: str, dataset name
        path: str, figure output path
        data_loader: pytorch dataloader
        z_tests: torch tensor (n_test, dim)
        influences: np array (n_test, n_train)
    """
    device = torch.device("cuda:{}".format(gpu))
    print("Computing training influence distribution by distance in latent space")

    influences = normalize(influences)

    # get latents
    B = 128
    latent_train = []
    with torch.no_grad():
        for i in range(len(data_loader) // B + 1):
            display_progress('Computing latents:', i * B, len(data_loader))
            if B * i == len(data_loader):
                continue
            if dataset[:5] in ['mnist', 'cifar']:
                x = torch.cat([data_loader.dataset[j][0].unsqueeze(0) for j in range(B*i, min(B*(i+1), len(data_loader)))])
            else:
                x = torch.cat([data_loader.dataset[j].unsqueeze(0) for j in range(B*i, min(B*(i+1), len(data_loader)))])
            latent_train.append(feat_net(x))
        latent_train = torch.cat(latent_train).cpu().numpy()
        latent_test = feat_net(torch.cat([x.unsqueeze(0) for x in z_tests])).cpu().numpy()

    # compute pairwise distance
    distances = np.zeros(influences.shape)
    for i in range(len(z_tests)):
        display_progress('Computing latent dist:', i, len(z_tests))
        for j in range(len(data_loader)):
            distances[i][j] = np.linalg.norm(latent_test[i] - latent_train[j])
    
    vec_distances, vec_influences = distances.reshape((-1,)), influences.reshape((-1,))
    
    # plot scatter
    from sklearn.preprocessing import minmax_scale
    fig, ax = plt.subplots()
    cmap = getattr(cm, 'plasma_r', cm.hot_r)
    ax.scatter(vec_distances, vec_influences, s=0.1, alpha=0.5, c=cmap(minmax_scale(np.abs(vec_influences))))
    ax.set_xlabel(r'dist($x_i$, $z_j$)')
    ax.set_ylabel(r'TracIn($x_i$, $z_j$)')
    plt.savefig(os.path.join(path, 'traininf_bydist.jpg'), dpi=400)


def get_inf_vs_latent_norm(feat_net, dataset, path, data_loader, z_tests, influences, gpu):
    """
    Plot distribution of influences vs norm in latent space

    Arguments:
        feat_net: torch neural network to compute latent
        dataset: str, dataset name
        path: str, figure output path
        data_loader: pytorch dataloader
        z_tests: torch tensor (n_test, dim)
        influences: np array (n_test, n_train)
        inception: bool, whether using inception v3 or own encoder
    """
    device = torch.device("cuda:{}".format(gpu))
    print("Computing training influence distribution by norm in latent space")

    influences = normalize(influences)

    # get latents
    B = 128
    latent_train = []
    with torch.no_grad():
        for i in range(len(data_loader) // B + 1):
            display_progress('Computing latents:', i * B, len(data_loader))
            if B * i == len(data_loader):
                continue
            if dataset[:5] in ['mnist', 'cifar']:
                x = torch.cat([data_loader.dataset[j][0].unsqueeze(0) for j in range(B*i, min(B*(i+1), len(data_loader)))])
            else:
                x = torch.cat([data_loader.dataset[j].unsqueeze(0) for j in range(B*i, min(B*(i+1), len(data_loader)))])
            latent_train.append(feat_net(x))
        latent_train = torch.cat(latent_train).cpu().numpy()
        latent_test = feat_net(torch.cat([x.unsqueeze(0) for x in z_tests])).cpu().numpy()
    
    # compute training sample norm
    norms = np.zeros(influences.shape)
    for j in range(len(data_loader)):
        display_progress('Computing training sample norm:', j, len(data_loader))
        _norm_ij = np.linalg.norm(latent_train[j])
        for i in range(len(z_tests)):
            norms[i][j] = _norm_ij

    vec_norms, vec_influences = norms.reshape((-1,)), influences.reshape((-1,))

    # plot scatter
    from sklearn.preprocessing import minmax_scale
    fig, ax = plt.subplots()
    cmap = getattr(cm, 'plasma_r', cm.hot_r)
    ax.scatter(vec_norms, vec_influences, s=0.1, alpha=0.5, c=cmap(minmax_scale(np.abs(vec_influences))))
    ax.set_xlabel(r'norm($x_i$)')
    ax.set_ylabel(r'TracIn($x_i$, $z_j$)')
    plt.savefig(os.path.join(path, 'traininf_bylatentnorm.jpg'), dpi=400)

