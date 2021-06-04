import os
import time
import numpy as np
import argparse
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from torch.autograd import grad

from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image, make_grid

from Beta_VAE.paths import *
from Beta_VAE.model import BetaVAE_H, BetaVAE_MNIST
from Beta_VAE.dataset import CustomImageFolder, CustomImageFolder_label
from tracin import loss_z, tracin_cp_all_self_inf

from utils import print_size
from statistics import get_auc_score, self_inf_distribution_by_elbo

import warnings
warnings.simplefilter("ignore")


def main(args):
    gpu = args.gpu
    device = torch.device("cuda:{}".format(gpu))
    print('-'*50)

    # define and load model
    if args.dataset == 'celeba':
        net = BetaVAE_H(z_dim=args.latent_dim)
        network_config = 'celeba_H_beta{}_z{}_sgd'.format(args.beta, args.latent_dim)
        nrow = 64
    elif args.dataset[:5] == 'cifar':
        net = BetaVAE_H(z_dim=args.latent_dim)
        network_config = '{}_beta{}_z{}_sgd'.format(args.dataset, args.beta, args.latent_dim)
        nrow = 64
    elif args.dataset[:5] == 'mnist':
        net = BetaVAE_MNIST(z_dim=args.latent_dim)
        network_config = '{}_beta{}_z{}_sgd'.format(args.dataset, args.beta, args.latent_dim)
        nrow = 28

    # load checkpoint
    checkpoint = torch.load(os.path.join('./Beta_VAE/checkpoints', network_config, str(args.ckpt_iter)),
                            map_location='cpu')
    net.load_state_dict(checkpoint['model_states']['net'])
    net = net.to(device)
    print_size(net)
    print('encoder has {} layers; decoder has {} layers'.format(len(list(net.encoder.parameters())), 
                                                                len(list(net.decoder.parameters()))))

    # define output path
    output_path = os.path.join('output', 'self_influence', network_config)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # data loader
    if args.dataset == 'celeba':
        transform = transforms.Compose([transforms.Resize((64, 64)),
                                        transforms.ToTensor(),])
        train_data = CustomImageFolder(ROOT_CELEBA, transform)

    elif args.dataset[:5] == 'cifar':
        transform = transforms.Compose([transforms.Resize((64, 64)),
                                        transforms.ToTensor()])
        train_data = datasets.CIFAR10(root=ROOT_CIFAR, transform=transform, download=True)

    elif args.dataset[:5] == 'mnist':
        transform = transforms.ToTensor()
        train_data = datasets.MNIST(root=ROOT_MNIST, transform=transform, download=True)

    numbers = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    for label, number in enumerate(numbers):
        if number in args.dataset.lower():
            number_ind = [i for i in range(len(train_data)) if train_data.targets[i] == label]
            train_data = Subset(train_data, number_ind)
            break
    
    # add a few samples as extra samples
    if 'emnist_extra' in args.dataset.lower():
        assert args.dataset.lower()[:5] == 'mnist'
        emnist_data = datasets.EMNIST(root=ROOT_EMNIST, 
                                      split='letters',
                                      transform=transforms.ToTensor(),
                                      download=True)
        emnist_data = Subset(emnist_data, range(1000))
        train_data = ConcatDataset([train_data, emnist_data])

    if 'celeba_extra' in args.dataset.lower():
        assert args.dataset.lower()[:5] == 'cifar'
        celeba_data = CustomImageFolder_label(root=ROOT_CELEBA,
                                              transform=transforms.Compose([transforms.Resize((64, 64)),
                                                                            transforms.ToTensor(),
                                                                            lambda x: (x - x.min()) / (x.max() - x.min())]))
        celeba_data = Subset(celeba_data, range(1000))
        train_data = ConcatDataset([train_data, celeba_data])

    data_loader = DataLoader(train_data,
                            batch_size=1,
                            shuffle=True,
                            num_workers=2,
                            pin_memory=True,
                            drop_last=True)
    print('dataloader prepared with {} samples'.format(len(train_data)))
    
    # compute/load VAE-TracIn self influences
    outfile = os.path.join(output_path, 'result{}{}.pkl'.format('_last{}'.format(args.use_last_layer) if args.use_last_layer>0 else '',
                                                                '_lastepoch' if args.last_epoch else ''))
    if not args.resume:
        config_dic = {'path': './Beta_VAE/checkpoints', 
                    'network_config': network_config, 
                    'suffix': '0000', 
                    'lr': 3e-4 if args.dataset[:5] == 'cifar' else 1e-4, 
                    'batchsize': 64,
                    'use_last_layer': args.use_last_layer}

        influences, harmful, helpful = tracin_cp_all_self_inf(args.dataset, args.latent_dim, data_loader, config_dic, 
                                                            args.reconstruct_num, args.beta, args.gpu, args.last_epoch)
        # save results
        dic = {'influences': influences,
               'harmful': harmful,
               'helpful': helpful}
        torch.save(dic, outfile)
    else:
        print('loading scores from {}'.format(outfile))
        dic = torch.load(outfile)
        influences, helpful, harmful = dic['influences'], dic['helpful'], dic['harmful']

    # unsupervised data cleaning application
    if 'extra' in args.dataset:
        if args.dataset[:5] == 'mnist':
            extra_indices = set(list(range(60000, 61000)))
        elif args.dataset[:5] == 'cifar':
            extra_indices = set(list(range(50000, 51000)))
        else:
            raise NotImplementedError

        flip_ind = list(extra_indices)
        auc, rates = get_auc_score(helpful, flip_ind, rescale=False)
        print('extra detection AUC (self-inf): {:1.4f}'.format(auc))

        plt.figure()
        plt.plot(np.linspace(0, 1, len(data_loader.dataset)), rates, label='TracIn self-inf')
        plt.xlabel('Fraction of training data checked')
        plt.ylabel('Fraction of extra data detected')
        plt.savefig(os.path.join(output_path, 'AUC.png'))

    # compute self influence distributions by loss
    loss_fn = lambda z: loss_z(z, net, args.gpu, args.beta, args.reconstruct_num)
    self_inf_distribution_by_elbo(output_path, args.dataset, data_loader, influences, loss_fn)

    # save most influential images
    X_helpful, X_harmful = [], []
    for i in range(args.n_display):
        if args.dataset == 'celeba':
            x_helpful = data_loader.dataset[helpful[i]]
            x_harmful = data_loader.dataset[harmful[i]]
        elif args.dataset[:5] == 'mnist' or args.dataset[:5] == 'cifar' :
            x_helpful = data_loader.dataset[helpful[i]][0]
            x_harmful = data_loader.dataset[harmful[i]][0]
            if 'emnist_extra' in args.dataset.lower():
                if helpful[i] in extra_indices:
                    x_helpful = x_helpful.permute(0,2,1)
                if harmful[i] in extra_indices:
                    x_harmful = x_harmful.permute(0,2,1)

        X_helpful.append(x_helpful)
        X_harmful.append(x_harmful)
        
    save_image(make_grid(X_helpful), fp=os.path.join(output_path, 'high_selfinf_train_samples.jpg'))
    save_image(make_grid(X_harmful), fp=os.path.join(output_path, 'los_selfinf_train_samples.jpg'))
    print('saved {} highest and lowest self influence samples'.format(args.n_display))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VAE-TracIn self influences')
    parser.add_argument('--gpu', default=0, type=int, help='gpu index')

    # model
    parser.add_argument('--beta', type=int, help='beta used to train model')
    parser.add_argument('--latent_dim', type=int, default=64, help='latent dimension')
    parser.add_argument('--ckpt_iter', type=int, default=1500000, help='checkpoint iteration')

    # dataset
    numbers = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    parser.add_argument('--dataset', type=str, choices=['mnist', 'mnist_emnist_extra', 'celeba', 
                                                        'cifar', 'cifar_celeba_extra'] + ['cifar_{}'.format(n) for n in numbers], 
                                               help='which dataset')
    
    # TracIn
    parser.add_argument('--resume', action='store_true', help='whether to use a pre-computed file')
    parser.add_argument('--use_last_layer', default=0, type=int, help='whether to only use the last a few layers to compute tracin')
    parser.add_argument('--last_epoch', action='store_true', help='whether to only use the last epoch checkpoint to compute tracin')
    parser.add_argument('--reconstruct_num', default=16, type=int, help='number of repeats of reconstruction to compute loss')

    # output
    parser.add_argument('--n_display', type=int, default=64, help='display this number of samples with the most positive/nagetive influences')

    args = parser.parse_args()
    main(args)
