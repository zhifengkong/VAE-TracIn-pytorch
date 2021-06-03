import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime as dt
import torch
from torch.nn import functional as F
import numpy as np


def tensor2image(x, size=(64, 64)):
    return x.cpu().detach().numpy().reshape(size)


def print_size(net):
    """
    Print the number of parameters of a network
    """
    if net is not None and isinstance(net, torch.nn.Module):
        module_parameters = filter(lambda p: p.requires_grad, net.parameters())
        params = sum([np.prod(p.size()) for p in module_parameters])
        print("{} Parameters: {:.6f}M".format(
            net.__class__.__name__, params / 1e6), flush=True)


class prettyfloat(float):
    def __repr__(self):
        return "%1.3g" % self


def generate(net, latent_dim, gpu):
    device = torch.device("cuda:{}".format(gpu))
    z = torch.normal(0.0, torch.ones(1, latent_dim))
    z = z.to(device)
    with torch.no_grad():
        x_recon = net._decode(z)
        x_gen = F.sigmoid(x_recon).data.squeeze(0)  # (nc, nrow, ncol)
    return x_gen


def reconstruct(net, data_loader, gen_num, gpu):
    device = torch.device("cuda:{}".format(gpu))
    x = torch.cat([data_loader.dataset[i][0].unsqueeze(0) for i in range(gen_num)]).to(device)
    with torch.no_grad():
        x_recon, _, _ = net(x)
        x_gen = F.sigmoid(x_recon).data  # (batchsize, nc, nrow, ncol)
    return x_gen


def fix_order(path):
    from copy import deepcopy
    all_pics = sorted(os.listdir(os.path.join(path, 'generated')))
    get_number = lambda x: int(x.split('.')[0])
    wrong_order = list(map(get_number, all_pics))
    dic = torch.load(os.path.join(path, 'result.pkl'))
    wrong_influences, wrong_helpful, wrong_harmful = dic['influences'], dic['helpful'], dic['harmful']
    influences, helpful, harmful = deepcopy(wrong_influences), deepcopy(wrong_helpful), deepcopy(wrong_harmful)
    for i, j in enumerate(wrong_order):
        # i = 2, j = 10, wrong_influences[2] -> influences[10]
        influences[j] = wrong_influences[i]
        helpful[j] = wrong_helpful[i]
        harmful[j] = wrong_harmful[i]
    torch.save({'influences': influences,
                'harmful': harmful,
                'helpful': helpful}, os.path.join(path, 'result.pkl'))


def concatenate_mnist(label1, label2, method):
    from torch.utils.data import Subset
    from torchvision import datasets, transforms
    from torchvision.utils import save_image

    root = '/tmp2/MNIST'
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(root=root, transform=transform, download=True)
    train_data = Subset(train_data, range(0, len(train_data), 10))

    x = torch.zeros(1, 28, 28)
    while True:
        i = np.random.randint(len(train_data))
        if train_data[i][1] == label1:
            break
    while True:
        j = np.random.randint(len(train_data))
        if train_data[j][1] == label2:
            break
    if method == 'average':
        x = (3 * train_data[i][0] + train_data[j][0]) / 4
    elif method == 'concatenate':
        x = x + train_data[i][0]
        x[:, 21:, :] = train_data[j][0][:, 21:, :]
    else:
        raise NotImplementedError
    save_image(x, fp=os.path.join('output/combined_jpg', 
                                  '{}_{}plus{}_from_{}and{}.jpg'.format(method, label1, label2, i, j)), 
               nrow=28, pad_value=1)


def get_number(name):
    name = name.split('.')[0]
    try:
        name = int(name)
    except:
        pass
    return name


def save_json(json_obj, json_path, append_if_exists=False,
              overwrite_if_exists=False, unique_fn_if_exists=True):
    """Saves a json file

    Arguments:
        json_obj: json, json object
        json_path: Path, path including the file name where the json object
            should be saved to
        append_if_exists: bool, append to the existing json file with the same
            name if it exists (keep the json structure intact)
        overwrite_if_exists: bool, xor with append, overwrites any existing
            target file
        unique_fn_if_exsists: bool, appends the current date and time to the
            file name if the target file exists already.
    """
    if isinstance(json_path, str):
        json_path = Path(json_path)

    if overwrite_if_exists:
        append_if_exists = False
        unique_fn_if_exists = False

    if unique_fn_if_exists:
        overwrite_if_exists = False
        append_if_exists = False
        if json_path.exists():
            time = dt.now().strftime("%Y-%m-%d-%H-%M-%S")
            json_path = json_path.parents[0] / f'{str(json_path.stem)}_{time}'\
                                               f'{str(json_path.suffix)}'

    if overwrite_if_exists:
        append_if_exists = False
        with open(json_path, 'w+') as fout:
            json.dump(json_obj, fout, indent=2)
        return

    if append_if_exists:
        if json_path.exists():
            with open(json_path, 'r') as fin:
                read_file = json.load(fin)
            read_file.update(json_obj)
            with open(json_path, 'w+') as fout:
                json.dump(read_file, fout, indent=2)
            return

    with open(json_path, 'w+') as fout:
        json.dump(json_obj, fout, indent=2)


def display_progress(text, current_step, last_step, enabled=True,
                     fix_zero_start=True):
    """Draws a progress indicator on the screen with the text preceeding the
    progress

    Arguments:
        test: str, text displayed to describe the task being executed
        current_step: int, current step of the iteration
        last_step: int, last possible step of the iteration
        enabled: bool, if false this function will not execute. This is
            for running silently without stdout output.
        fix_zero_start: bool, if true adds 1 to each current step so that the
            display starts at 1 instead of 0, which it would for most loops
            otherwise.
    """
    if not enabled:
        return

    # Fix display for most loops which start with 0, otherwise looks weird
    if fix_zero_start:
        current_step = current_step + 1

    term_line_len = 80
    final_chars = [':', ';', ' ', '.', ',']
    if text[-1:] not in final_chars:
        text = text + ' '
    if len(text) < term_line_len:
        bar_len = term_line_len - (len(text)
                                   + len(str(current_step))
                                   + len(str(last_step))
                                   + len("  / "))
    else:
        bar_len = 30
    filled_len = int(round(bar_len * current_step / float(last_step)))
    bar = '=' * filled_len + '.' * (bar_len - filled_len)

    bar = f"{text}[{bar:s}] {current_step:d} / {last_step:d}"
    if current_step < last_step-1:
        # Erase to end of line and print
        sys.stdout.write("\033[K" + bar + "\r")
    else:
        sys.stdout.write(bar + "\n")

    sys.stdout.flush()


def init_logging(filename=None):
    """Initialises log/stdout output

    Arguments:
        filename: str, a filename can be set to output the log information to
            a file instead of stdout"""
    log_lvl = logging.INFO
    log_format = '%(asctime)s: %(message)s'
    if filename:
        logging.basicConfig(handlers=[logging.FileHandler(filename),
                                      logging.StreamHandler(sys.stdout)],
                            level=log_lvl,
                            format=log_format)
    else:
        logging.basicConfig(stream=sys.stdout, level=log_lvl,
                            format=log_format)


def get_default_config():
    """Returns a default config file"""
    config = {
        'outdir': 'outdir',
        'seed': 42,
        'gpu': 0,
        'dataset': 'CIFAR10',
        'num_classes': 10,
        'test_sample_num': 1,
        'test_start_index': 0,
        'recursion_depth': 1,
        'r_averaging': 1,
        'scale': None,
        'damp': None,
        'calc_method': 'img_wise',
        'log_filename': None,
    }

    return config
