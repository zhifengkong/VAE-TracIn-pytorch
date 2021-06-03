import os
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms

class CustomImageFolder_label(ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder_label, self).__init__(root, transform)

    def __getitem__(self, index):
        path = self.imgs[index][0]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return (img, 10)

batch_size = 4
params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True}
transform = transforms.Compose([transforms.Resize([64, 64]),
                                transforms.ToTensor()])
train_kwargs = {'root': '/tmp2/CIFAR', 'transform': transform, 'download': True}
train_data = torchvision.datasets.CIFAR10(**train_kwargs)

celeba_data = CustomImageFolder_label(root='/tmp2/celebA/CelebA64/CelebA',
                                    transform=transforms.Compose([transforms.ToTensor(),
                                                                lambda x: (x - x.min()) / (x.max() - x.min())]))
                                                                #lambda x: torch.tensor(255 * x, dtype=torch.uint8())]))
celeba_data = Subset(celeba_data, range(1000))
train_data = ConcatDataset([train_data, celeba_data])

train_loader = DataLoader(train_data,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=1,
                        pin_memory=True,
                        drop_last=True)

for data in train_loader:
    X, y = data
    if 10 in y:
        print(X, y)
        break