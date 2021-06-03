import os
import numpy as np

import torch
from torch.autograd import grad
import torch.nn as nn
from torch.nn import functional as F
from utils import display_progress

from Beta_VAE.model import BetaVAE_H, BetaVAE_MNIST

import warnings
warnings.simplefilter("ignore")


def get_ordered_checkpoint_list(path, network_config, suffix='', last_epoch=False):
    all_files = os.listdir(os.path.join(path, network_config))
    all_checkpoints = [f for f in all_files if f.endswith(suffix)]
    all_checkpoints.sort(key=int)
    print('{} checkpoints found'.format(len(all_checkpoints)))
    if not last_epoch:
        return all_checkpoints
    else:
        print('using the last epoch only')
        return [all_checkpoints[-1]]


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


def calc_loss(x, x_recon, mu, logvar, beta):
    """Calculates the loss

    Arguments:
        x: torch tensor, input with size (minibatch, 3, 64, 64)
        x_recon: torch tensor, reconstructed x of size (minibatch, 3, 64, 64)

    Returns:
        loss: scalar, the loss"""
    batch_size = x.size(0)
    assert batch_size != 0

    x_recon = torch.sigmoid(x_recon)
    recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)
    total_kld, _, _ = kl_divergence(mu, logvar)
    beta_vae_loss = recon_loss + beta * total_kld
    return beta_vae_loss / (3 * 64 * 64)


def loss_z(z, model, gpu, beta, reconstruct_num):
    """Calculates the gradient z. One grad_z should be computed for each
    training sample.

    Arguments:
        z: torch tensor, a training data point
            e.g. an image sample (3, 64, 64)
        model: torch NN, model used to evaluate the dataset
        gpu: int, device id to use for GPU, -1 for CPU
        beta: int, betaVAE loss
        reconstruct_num: int, average the loss this times

    Returns:
        loss_z
    """
    model.eval()
    # initialize
    if gpu >= 0:
        device = torch.device("cuda:{}".format(gpu))
        z = z.to(device)
    
    with torch.no_grad():
        loss = 0.0
        for _ in range(reconstruct_num):
            z_recon, mu, logvar = model(z.unsqueeze(0))
            loss += calc_loss(z, z_recon, mu, logvar, beta=beta)
        loss /= reconstruct_num
    return loss.cpu().item()


def grad_z_last_layer(z, model, gpu, beta, reconstruct_num, use_last_layer):
    """Calculates the gradient z. One grad_z should be computed for each
    training sample.

    Arguments:
        z: torch tensor, training data points
            e.g. an image sample (batch_size, 3, 256, 256)
        model: torch NN, model used to evaluate the dataset
        gpu: int, device id to use for GPU, -1 for CPU
        beta: int, betaVAE loss
        reconstruct_num: int, average the loss this times
        use_last_layer: int, whether to compute gradient for last a few layers

    Returns:
        grad_z: list of torch tensor, containing the gradients
            from model parameters to loss"""
    model.eval()
    # initialize
    if gpu >= 0:
        device = torch.device("cuda:{}".format(gpu))
        z = z.to(device)
    
    loss = 0.0
    for _ in range(reconstruct_num):
        z_recon, mu, logvar = model(z.unsqueeze(0))
        loss += calc_loss(z, z_recon, mu, logvar, beta=beta)
    loss /= reconstruct_num
    
    # Compute sum of gradients from model parameters to loss
    params_encoder = [ p for p in model.encoder.parameters() if p.requires_grad ]
    params_decoder = [ p for p in model.decoder.parameters() if p.requires_grad ]
    if use_last_layer > 0:
        params = params_encoder[-use_last_layer:] + params_decoder[-use_last_layer:]
    else:
        params = params_encoder + params_decoder
    return list(grad(loss, params, create_graph=True))


def tracin_cp(dataset, latent_dim, data_loader, z_test, dic, reconstruct_num, beta, gpu, last_epoch):
    """
    dic.keys(): path, network_config, suffix, lr, batchsize, use_last_layer
    """

    train_dataset_size = len(data_loader.dataset)
    device = torch.device("cuda:{}".format(gpu))
    z_test = z_test.to(device)

    ckpt_iter = 0
    influences = [0.0 for _ in range(train_dataset_size)]
    ordered_checkpoint_list = get_ordered_checkpoint_list(dic['path'], dic['network_config'], dic['suffix'], last_epoch)

    # add \nabla loss(z_i, beta) * \nabla loss(z_test, beta) for every z_i over all steps
    for checkpoint_name in ordered_checkpoint_list:
        # load model at this checkpoint
        if dataset == 'celeba' or dataset[:5] == 'cifar':
            model = BetaVAE_H(z_dim=latent_dim)
        elif dataset[:5] == 'mnist':
            model = BetaVAE_MNIST(z_dim=latent_dim)
        checkpoint = torch.load(os.path.join(dic['path'], dic['network_config'], checkpoint_name),
                                map_location='cpu')
        model.load_state_dict(checkpoint['model_states']['net'])
        model = model.to(device)
        
        # print checkpoint iteration
        ckpt_interval = int(checkpoint_name) - ckpt_iter
        ckpt_iter = int(checkpoint_name)
        print('checkpoint: {}'.format(ckpt_iter))

        # compute gradient at z_test
        grad_test = grad_z_last_layer(z_test, model, gpu, beta, reconstruct_num, dic['use_last_layer'])

        # compute gradient at z_i for each i
        for i in range(train_dataset_size):
            if dataset == 'celeba':
                z_train = data_loader.dataset[i].to(device)
            elif dataset[:5] == 'mnist' or dataset[:5] == 'cifar':
                z_train = data_loader.dataset[i][0].to(device)
            grad_train = grad_z_last_layer(z_train, model, gpu, beta, reconstruct_num, dic['use_last_layer'])
        
            # add gradient dot product to influences
            grad_dot_product = sum([torch.sum(k * j).data for k, j in zip(grad_train, grad_test)]).cpu().numpy()
            influences[i] += grad_dot_product * dic['lr'] * dic['batchsize'] * ckpt_interval / train_dataset_size
            
            display_progress("Calc. grad dot product: ", i, train_dataset_size)
    
    influences = np.array(influences)
    harmful = np.argsort(influences)
    helpful = harmful[::-1]
    return influences, harmful.tolist(), helpful.tolist()


def tracin_cp_multiple(dataset, latent_dim, data_loader, z_tests, dic, reconstruct_num, beta, gpu, last_epoch):
    """
    dic.keys(): path, network_config, suffix, lr, batchsize, use_last_layer
    """

    train_dataset_size = len(data_loader.dataset)
    device = torch.device("cuda:{}".format(gpu))

    ckpt_iter = 0
    influences = [[0.0 for _ in range(train_dataset_size)] for _ in z_tests]
    ordered_checkpoint_list = get_ordered_checkpoint_list(dic['path'], dic['network_config'], dic['suffix'], last_epoch)

    # add \nabla loss(z_i, beta) * \nabla loss(z_test, beta) for every z_i over all steps
    for checkpoint_name in ordered_checkpoint_list:
        # load model at this checkpoint
        if dataset == 'celeba' or dataset[:5] == 'cifar':
            model = BetaVAE_H(z_dim=latent_dim)
        elif dataset[:5] == 'mnist':
            model = BetaVAE_MNIST(z_dim=latent_dim)
        checkpoint = torch.load(os.path.join(dic['path'], dic['network_config'], checkpoint_name),
                                map_location='cpu')
        model.load_state_dict(checkpoint['model_states']['net'])
        model = model.to(device)
        
        # print checkpoint iteration
        ckpt_interval = int(checkpoint_name) - ckpt_iter
        ckpt_iter = int(checkpoint_name)
        print('checkpoint: {}'.format(ckpt_iter))

        # compute gradient at z_test
        grad_tests = [grad_z_last_layer(z_test.to(device), 
                                        model, gpu, beta, 
                                        reconstruct_num, 
                                        dic['use_last_layer']) for z_test in z_tests]

        # compute gradient at z_i for each i
        for i in range(train_dataset_size):
            if dataset == 'celeba':
                z_train = data_loader.dataset[i].to(device)
            elif dataset[:5] == 'mnist' or dataset[:5] == 'cifar':
                z_train = data_loader.dataset[i][0].to(device)
            grad_train = grad_z_last_layer(z_train, model, gpu, beta, reconstruct_num, dic['use_last_layer'])
        
            # add gradient dot product to influences
            for j in range(len(z_tests)):
                grad_dot_product = sum([torch.sum(k * j).data for k, j in zip(grad_train, grad_tests[j])]).cpu().numpy()
                influences[j][i] += grad_dot_product * dic['lr'] * dic['batchsize'] * ckpt_interval / train_dataset_size
            
            display_progress("Calc. grad dot product: ", i, train_dataset_size)
    
    influences = np.array(influences)
    harmful = np.array([np.argsort(influence) for influence in influences])
    helpful = np.array([x[::-1] for x in harmful])
    return influences, harmful.tolist(), helpful.tolist()


def tracin_cp_all_self_inf(dataset, latent_dim, data_loader, dic, reconstruct_num, beta, gpu, last_epoch):
    """
    dic.keys(): path, network_config, suffix, lr, batchsize, use_last_layer
    """

    train_dataset_size = len(data_loader.dataset)
    device = torch.device("cuda:{}".format(gpu))

    ckpt_iter = 0
    self_influences = [0.0 for _ in range(train_dataset_size)]
    ordered_checkpoint_list = get_ordered_checkpoint_list(dic['path'], dic['network_config'], dic['suffix'], last_epoch)

    # add \nabla loss(z_i, beta) * \nabla loss(z_i, beta) for every z_i over all steps
    for checkpoint_name in ordered_checkpoint_list:
        # load model at this checkpoint
        if dataset == 'celeba' or dataset[:5] == 'cifar':
            model = BetaVAE_H(z_dim=latent_dim)
        elif dataset[:5] == 'mnist':
            model = BetaVAE_MNIST(z_dim=latent_dim)
        checkpoint = torch.load(os.path.join(dic['path'], dic['network_config'], checkpoint_name),
                                map_location='cpu')
        model.load_state_dict(checkpoint['model_states']['net'])
        model = model.to(device)
        
        # print checkpoint iteration
        ckpt_interval = int(checkpoint_name) - ckpt_iter
        ckpt_iter = int(checkpoint_name)
        print('checkpoint: {}'.format(ckpt_iter))

        # compute gradient at z_i for each i
        for i in range(train_dataset_size):
            if dataset == 'celeba':
                z_train = data_loader.dataset[i].to(device)
            elif dataset[:5] == 'mnist' or dataset[:5] == 'cifar':
                z_train = data_loader.dataset[i][0].to(device)
            grad_train = grad_z_last_layer(z_train, model, gpu, beta, reconstruct_num, dic['use_last_layer'])
        
            # add gradient dot product to self influences
            grad_dot_product = sum([torch.sum(k * k).data for k in grad_train]).cpu().numpy()
            self_influences[i] += grad_dot_product * dic['lr'] * dic['batchsize'] * ckpt_interval / train_dataset_size
            
            display_progress("Calc. grad dot product: ", i, train_dataset_size)
    
    self_influences = np.array(self_influences)
    self_harmful = np.argsort(self_influences)
    self_helpful = self_harmful[::-1]
    return self_influences, self_harmful.tolist(), self_helpful.tolist()
