"""utils.py"""

import argparse
import subprocess

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np


def cuda(tensor, uses_cuda, gpu):
    if not uses_cuda:
        return tensor
    device = torch.device("cuda:{}".format(gpu))
    return tensor.to(device)


def str2bool(v):
    # codes from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def where(cond, x, y):
    """Do same operation as np.where

    code from:
        https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/8
    """
    cond = cond.float()
    return (cond*x) + ((1-cond)*y)


def grid2gif(image_str, output_gif, delay=100):
    """Make GIF from images.

    code from:
        https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python/34555939#34555939
    """
    str1 = 'convert -delay '+str(delay)+' -loop 0 ' + image_str  + ' ' + output_gif
    subprocess.call(str1, shell=True)


def print_size(net):
    """
    Print the number of parameters of a network
    """

    if net is not None and isinstance(net, torch.nn.Module):
        module_parameters = filter(lambda p: p.requires_grad, net.parameters())
        params = sum([np.prod(p.size()) for p in module_parameters])
        print("{} Parameters: {:.6f}M".format(
            net.__class__.__name__, params / 1e6), flush=True)