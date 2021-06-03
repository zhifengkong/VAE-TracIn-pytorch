import os
import time
import numpy as np
import argparse
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.autograd import grad

from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image, make_grid

from Beta_VAE.paths import *
from Beta_VAE.model import BetaVAE_H, BetaVAE_MNIST
from Beta_VAE.dataset import CustomImageFolder
from tracin import tracin_cp_multiple

from utils import print_size, generate, get_number
from statistics import train_inf_distribution_by_label, get_inf_vs_latent_dist, get_inf_vs_latent_norm

import warnings
warnings.filterwarnings("ignore")

ROOT_MNIST = '/tmp2/MNIST'
ROOT_CIFAR = '/tmp2/CIFAR'
ROOT_CELEBA = '/tmp2/celebA/CelebA64/CelebA'


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
    output_path = os.path.join('output', args.task, network_config)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # data loader
    if args.dataset == 'celeba':
        transform = transforms.Compose([transforms.Resize((64, 64)),
                                        transforms.ToTensor(),])
        train_data = CustomImageFolder(ROOT_CELEBA, transform)
        test_data = None

    elif args.dataset[:5] == 'cifar':
        transform = transforms.Compose([transforms.Resize((64, 64)),
                                        transforms.ToTensor()])
        train_data = datasets.CIFAR10(root=ROOT_CIFAR, transform=transform, download=True)
        test_data = datasets.CIFAR10(root=ROOT_CIFAR, transform=transform, download=True, train=False)

    elif args.dataset[:5] == 'mnist':
        transform = transforms.ToTensor()
        train_data = datasets.MNIST(root=ROOT_MNIST, transform=transform, download=True)
        test_data = datasets.MNIST(root=ROOT_MNIST, transform=transform, download=True, train=False)
    
    numbers = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    for label, number in enumerate(numbers):
        if number in args.dataset.lower():
            number_ind = [i for i in range(len(train_data)) if train_data.targets[i] == label]
            train_data = Subset(train_data, number_ind)
            break

    data_loader = DataLoader(train_data,
                            batch_size=1,
                            shuffle=True,
                            num_workers=2,
                            pin_memory=True,
                            drop_last=True)
    print('data_loader prepared with {} samples'.format(len(train_data)))

    # read z_test
    if args.load_from_jpg:
        from PIL import Image, ImageOps

        if args.load_from_jpg_dir:
            z_test_dir = args.load_from_jpg_dir
        else:
            z_test_dir = os.path.join(output_path, 'test')
        
        z_tests = []
        all_pics = sorted(os.listdir(z_test_dir), key=get_number)
        for pic in all_pics:
            image = Image.open(os.path.join(z_test_dir, pic))
            if args.dataset[:5] == 'mnist':
                image = ImageOps.grayscale(image)
            z_tests.append(transform(image))
        save_image(make_grid(z_tests), fp=os.path.join(output_path, 'test.jpg'))

    # generate z_test
    else:
        # load original model and do sampling
        if args.dataset[:5] == 'mnist':
            net_to_generate = BetaVAE_MNIST(z_dim=args.latent_dim)
        elif args.dataset[:5] == 'cifar':
            net_to_generate = BetaVAE_H(z_dim=args.latent_dim)
        checkpoint = torch.load(os.path.join('./Beta_VAE/checkpoints', network_config, str(args.ckpt_iter)),
                            map_location='cpu')
        net_to_generate.load_state_dict(checkpoint['model_states']['net'])
        net_to_generate = net_to_generate.to(device)
        del checkpoint

        if args.task == 'gen':
            z_tests = [generate(net_to_generate, 
                                args.latent_dim, 
                                args.gpu) for _ in range(args.test_num)]
        elif args.task == 'train':
            z_tests = torch.cat([data_loader.dataset[i][0].unsqueeze(0) for i in range(args.test_num)])
        save_image(make_grid(z_tests), fp=os.path.join(output_path, 'test.jpg'))

        if 'test' not in os.listdir(output_path):
            os.mkdir(os.path.join(output_path, 'test'))
        for i, x_gen in enumerate(z_tests):
            save_image(x_gen, fp=os.path.join(output_path, 'test', '{}.jpg'.format(i)), nrow=nrow, pad_value=1)

        print('obtained {} test samples'.format(len(z_tests)))

    # compute/load VAE-TracIn test data influences
    outfile = os.path.join(output_path, 'result{}{}.pkl'.format('_last{}'.format(args.use_last_layer) if args.use_last_layer>0 else '',
                                                                '_lastepoch' if args.last_epoch else ''))
    if not args.resume:
        config_dic = {'path': './Beta_VAE/checkpoints', 
                    'network_config': network_config, 
                    'suffix': '0000', 
                    'lr': 3e-4 if args.dataset[:5] == 'cifar' else 1e-4, 
                    'batchsize': 64,
                    'use_last_layer': args.use_last_layer}
        influences, harmful, helpful = tracin_cp_multiple(args.dataset, args.latent_dim, data_loader, z_tests, config_dic, 
                                                    args.reconstruct_num, args.beta, args.gpu, args.last_epoch)
        # save results
        dic = {'influences': influences,
               'harmful': harmful,
               'helpful': helpful}
        torch.save(dic, outfile)
    else:
        print('loading tracin scores from {}'.format(outfile))
        dic = torch.load(outfile)
        influences, helpful, harmful = dic['influences'], dic['helpful'], dic['harmful']

    # sanity check
    if args.task == 'train':
        tracin_rank = []
        for j in range(args.test_num):
            tracin_rank.append(helpful[j].index(j))                                                   
        print('Top-1 Acc.:', len(list(filter(lambda x: x == 0, tracin_rank))) / len(tracin_rank))

    # influence distribution
    if args.task == 'test':
        # influence versus label
        if args.dataset[:5] == 'mnist':
            test_data_loader = DataLoader(test_data, batch_size=1)
            test_labels = [int(test_data_loader.dataset[int(pic.split('.')[0].split('_')[1])][1]) for pic in all_pics]
            train_inf_distribution_by_label(output_path, data_loader, list(range(10)), test_labels, influences)

        # influence versus latent distance
        if args.dataset[:5] == 'mnist':
            feat_net = lambda x: net._encode(x.to(device))[:, :args.latent_dim]
            get_inf_vs_latent_dist(feat_net, args.dataset, output_path, data_loader, z_tests, influences, args.gpu)

        # influence versus latent norm
        if args.dataset[:5] == 'cifar':
            feat_net = lambda x: net._encode(x.to(device))[:, :args.latent_dim]
            get_inf_vs_latent_norm(feat_net, args.dataset, output_path, data_loader, z_tests, influences, args.gpu)

    # save most influential images
    X_helpful, X_harmful,  = [], []
    for j in range(len(z_tests)):
        for i in range(args.n_display):
            if args.dataset == 'celeba':
                x_helpful = data_loader.dataset[helpful[j][i]]
                x_harmful = data_loader.dataset[harmful[j][i]]
            elif args.dataset[:5] == 'mnist' or args.dataset[:5] == 'cifar':
                x_helpful = data_loader.dataset[helpful[j][i]][0]
                x_harmful = data_loader.dataset[harmful[j][i]][0]
            X_helpful.append(x_helpful)
            X_harmful.append(x_harmful)
    save_image(make_grid(X_helpful, nrow=args.n_display), fp=os.path.join(output_path, 'proponents.jpg'))
    save_image(make_grid(X_harmful, nrow=args.n_display), fp=os.path.join(output_path, 'opponents.jpg'))
    print('saved {} most/least influential samples for {} images'.format(args.n_display, len(z_tests)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VAE-TracIn self influences')
    parser.add_argument('--gpu', default=0, type=int, help='gpu index')

    # model
    parser.add_argument('--beta', type=int, help='beta used to train model')
    parser.add_argument('--latent_dim', type=int, default=64, help='latent dimension')
    parser.add_argument('--ckpt_iter', type=int, default=1500000, help='checkpoint iteration')

    # dataset
    numbers = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    parser.add_argument('--dataset', type=str, choices=['mnist', 'celeba', 'cifar'] + ['cifar_{}'.format(n) for n in numbers], 
                                               help='which dataset')
    parser.add_argument('--test_num', type=int, default=128, help='number of generated images')
    parser.add_argument('--load_from_jpg', action='store_true', help='whether to use a pre-computed z_test')
    parser.add_argument('--load_from_jpg_dir', type=str, default='', help='dir of the pre-computed z_test')
    
    # TracIn
    parser.add_argument('--task', choices=['train', 'test', 'gen'], type=str, help='task')
    parser.add_argument('--resume', action='store_true', help='whether to use a pre-computed file')
    parser.add_argument('--use_last_layer', default=0, type=int, help='whether to only use the last a few layers to compute tracin')
    parser.add_argument('--last_epoch', action='store_true', help='whether to only use the last epoch checkpoint to compute tracin')
    parser.add_argument('--reconstruct_num', default=16, type=int, help='number of repeats of reconstruction to compute loss')

    # output
    parser.add_argument('--n_display', type=int, default=8, help='display this number of samples with the most positive/nagetive influences')

    args = parser.parse_args()
    main(args)
