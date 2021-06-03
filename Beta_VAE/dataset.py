"""dataset.py"""

import os
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms

from paths import *


def is_power_of_2(num):
    return ((num & (num - 1)) == 0) and num != 0


class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)

    def __getitem__(self, index):
        path = self.imgs[index][0]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img

class CustomImageFolder_label(ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder_label, self).__init__(root, transform)

    def __getitem__(self, index):
        path = self.imgs[index][0]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return (img, 0)


class CustomTensorDataset(Dataset):
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __getitem__(self, index):
        return self.data_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)


def return_data(args):
    name = args.dataset
    dset_dir = args.dset_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    image_size = args.image_size
    # assert image_size == 64, 'currently only image size of 64 is supported'

    if name.lower() == '3dchairs':
        root = os.path.join(dset_dir, '3DChairs')
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),])
        train_kwargs = {'root':root, 'transform':transform}
        dset = CustomImageFolder

    elif name.lower() == 'celeba':
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),])
        train_kwargs = {'root': ROOT_CELEBA, 'transform':transform}
        dset = CustomImageFolder

    elif name.lower() == 'dsprites':
        root = os.path.join(dset_dir, 'dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        if not os.path.exists(root):
            import subprocess
            print('Now download dsprites-dataset')
            subprocess.call(['./download_dsprites.sh'])
            print('Finished')
        data = np.load(root, encoding='bytes')
        data = torch.from_numpy(data['imgs']).unsqueeze(1).float()
        train_kwargs = {'data_tensor':data}
        dset = CustomTensorDataset

    elif name.lower()[:5] == 'mnist':
        train_kwargs = {'root': ROOT_MNIST, 'transform': transforms.ToTensor(), 'download': True}
        dset = torchvision.datasets.MNIST

    elif name.lower()[:5] == 'cifar':
        params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True}
        transform = transforms.Compose([transforms.Resize([image_size, image_size]),
                                        transforms.ToTensor()])
        train_kwargs = {'root': ROOT_CIFAR, 'transform': transform, 'download': True}
        dset = torchvision.datasets.CIFAR10

    else:
        raise NotImplementedError
    
    train_data = dset(**train_kwargs)
    
    numbers = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    for label, number in enumerate(numbers):
        if number in name.lower():
            number_ind = [i for i in range(len(train_data)) if train_data.targets[i] == label]
            train_data = Subset(train_data, number_ind)
            break

    # add extra samples
    if 'emnist_extra' in name.lower():
        assert name.lower()[:5] == 'mnist'
        emnist_data = torchvision.datasets.EMNIST(root='/tmp2/EMNIST', 
                                                split='letters', 
                                                transform=transforms.ToTensor(),
                                                download=True)
        emnist_data = Subset(emnist_data, range(1000))
        train_data = ConcatDataset([train_data, emnist_data])

    if 'celeba_extra' in name.lower():
        assert name.lower()[:5] == 'cifar'
        celeba_data = CustomImageFolder_label(root='/tmp2/celebA/CelebA64/CelebA',
                                              transform=transforms.Compose([transforms.Resize((image_size, image_size)),
                                                                            transforms.ToTensor(),
                                                                            lambda x: (x - x.min()) / (x.max() - x.min())]))
        celeba_data = Subset(celeba_data, range(1000))
        train_data = ConcatDataset([train_data, celeba_data])

    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=True)
    data_loader = train_loader
    print('loaded dataset with {} samples'.format(len(train_data)))
    return data_loader

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),])

    dset = CustomImageFolder('data/CelebA', transform)
    loader = DataLoader(dset,
                       batch_size=32,
                       shuffle=True,
                       num_workers=1,
                       pin_memory=False,
                       drop_last=True)

    images1 = iter(loader).next()
    import ipdb; ipdb.set_trace()
