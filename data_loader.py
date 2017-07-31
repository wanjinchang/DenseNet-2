import os
import numpy as np
from utils import plot_images

import torch
from torchvision import datasets
from torchvision import transforms

def get_loader(data_dir, 
               is_train, 
               batch_size,
               augment,
               num_workers=4,
               pin_memory=False, 
               shuffle=True, 
               show_sample=False):
    """
    Utility function for loading and returning a multi-process iterator
    over the CIFAR-10 dataset. A sample 9x9 grid of the images can be
    optionally displayed.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Params
    ------
    - data_dir: path directory to the dataset.
    - is_train: whether to return an iterator over the train set. If set
      to false, returns one for the test set.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper.
    - num_workers: number of subprocessed to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    - shuffle: whether to shuffle the dataset after every epoch.
    - show_sample: plot 9x9 sample grid of the dataset.
    """

    if augment:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

    dataset = datasets.CIFAR10(root=data_dir, 
                               train=is_train, 
                               download=True,
                               transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset, 
                                             batch_size=batch_size, 
                                             shuffle=shuffle, 
                                             num_workers=num_workers,
                                             pin_memory=pin_memory)

    # visualize some images
    if show_sample:
        sample_loader = torch.utils.data.DataLoader(dataset, 
                                                    batch_size=9, 
                                                    shuffle=shuffle, 
                                                    num_workers=num_workers,
                                                    pin_memory=pin_memory)
        data_iter = iter(sample_loader)
        images, labels = data_iter.next()
        X = images.numpy()
        X = np.transpose(X, [0, 2, 3, 1])
        plot_images(X, labels)

    return data_loader