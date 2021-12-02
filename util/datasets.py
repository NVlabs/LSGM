# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for LSGM. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

"""Code for getting the data loaders."""

import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from util.lmdb_datasets import LMDBDataset
from thirdparty.lsun import LSUN, LSUNClass
import os
import urllib
from scipy.io import loadmat
from torch.utils.data import Dataset
from PIL import Image
from torch._utils import _accumulate

class Binarize(object):
    """ This class introduces a binarization transformation
    """
    def __call__(self, pic):
        return torch.Tensor(pic.size()).bernoulli_(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class CropCelebA64(object):
    """ This class applies cropping for CelebA64. This is a simplified implementation of:
    https://github.com/andersbll/autoencoding_beyond_pixels/blob/master/dataset/celeba.py
    """
    def __call__(self, pic):
        new_pic = pic.crop((15, 40, 178 - 15, 218 - 30))
        return new_pic

    def __repr__(self):
        return self.__class__.__name__ + '()'


def get_loaders(args):
    """Get data loaders for required dataset."""
    return get_loaders_eval(args.dataset, args.data, args.distributed, args.batch_size)


def download_omniglot(data_dir):
    filename = 'chardata.mat'
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    url = 'https://raw.github.com/yburda/iwae/master/datasets/OMNIGLOT/chardata.mat'

    filepath = os.path.join(data_dir, filename)
    if not os.path.exists(filepath):
        filepath, _ = urllib.request.urlretrieve(url, filepath)
        print('Downloaded', filename)

    return


def load_omniglot(data_dir):
    download_omniglot(data_dir)

    data_path = os.path.join(data_dir, 'chardata.mat')

    omni = loadmat(data_path)
    train_data = 255 * omni['data'].astype('float32').reshape((28, 28, -1)).transpose((2, 1, 0))
    test_data = 255 * omni['testdata'].astype('float32').reshape((28, 28, -1)).transpose((2, 1, 0))

    train_data = train_data.astype('uint8')
    test_data = test_data.astype('uint8')

    return train_data, test_data


class OMNIGLOT(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        d = self.data[index]
        img = Image.fromarray(d)
        return self.transform(img), 0     # return zero as label.

    def __len__(self):
        return len(self.data)


def get_loaders_eval(dataset, root, distributed, batch_size, augment=True, drop_last_train=True, shuffle_train=True,
                     binarize_binary_datasets=True):
    if dataset == 'cifar10':
        num_classes = 10
        train_transform, valid_transform = _data_transforms_cifar10()
        train_transform = train_transform if augment else valid_transform
        train_data = dset.CIFAR10(
            root=root, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR10(
            root=root, train=False, download=True, transform=valid_transform)
    elif dataset == 'mnist':
        num_classes = 10
        train_transform, valid_transform = _data_transforms_mnist(binarize_binary_datasets)
        train_transform = train_transform if augment else valid_transform
        train_data = dset.MNIST(
            root=root, train=True, download=True, transform=train_transform)
        valid_data = dset.MNIST(
            root=root, train=False, download=True, transform=valid_transform)
    elif dataset == 'omniglot':
        num_classes = 0
        download_omniglot(root)
        train_transform, valid_transform = _data_transforms_mnist(binarize_binary_datasets)
        train_transform = train_transform if augment else valid_transform
        train_data, valid_data = load_omniglot(root)
        train_data = OMNIGLOT(train_data, train_transform)
        valid_data = OMNIGLOT(valid_data, valid_transform)
    elif dataset.startswith('celeba'):
        if dataset == 'celeba_64':
            resize = 64
            num_classes = 40
            train_transform, valid_transform = _data_transforms_celeba64(resize)
            train_transform = train_transform if augment else valid_transform
            train_data = LMDBDataset(root=root, name='celeba64', train=True, transform=train_transform, is_encoded=True)
            valid_data = LMDBDataset(root=root, name='celeba64', train=False, transform=valid_transform, is_encoded=True)
        elif dataset in {'celeba_256'}:
            num_classes = 1
            resize = int(dataset.split('_')[1])
            train_transform, valid_transform = _data_transforms_generic(resize)
            train_transform = train_transform if augment else valid_transform
            train_data = LMDBDataset(root=root, name='celeba', train=True, transform=train_transform)
            valid_data = LMDBDataset(root=root, name='celeba', train=False, transform=valid_transform)
        else:
            raise NotImplementedError
    elif dataset.startswith('lsun'):
        if dataset.startswith('lsun_bedroom'):
            resize = int(dataset.split('_')[-1])
            num_classes = 1
            train_transform, valid_transform = _data_transforms_lsun(resize)
            train_transform = train_transform if augment else valid_transform
            train_data = LSUN(root=root, classes=['bedroom_train'], transform=train_transform)
            valid_data = LSUN(root=root, classes=['bedroom_val'], transform=valid_transform)
        elif dataset.startswith('lsun_church'):
            resize = int(dataset.split('_')[-1])
            num_classes = 1
            train_transform, valid_transform = _data_transforms_lsun(resize)
            train_transform = train_transform if augment else valid_transform
            train_data = LSUN(root=root, classes=['church_outdoor_train'], transform=train_transform)
            valid_data = LSUN(root=root, classes=['church_outdoor_val'], transform=valid_transform)
        elif dataset.startswith('lsun_tower'):
            resize = int(dataset.split('_')[-1])
            num_classes = 1
            train_transform, valid_transform = _data_transforms_lsun(resize)
            train_transform = train_transform if augment else valid_transform
            train_data = LSUN(root=root, classes=['tower_train'], transform=train_transform)
            valid_data = LSUN(root=root, classes=['tower_val'], transform=valid_transform)
        elif dataset.startswith('lsun_cat'):
            resize = int(dataset.split('_')[-1])
            num_classes = 1
            train_transform, valid_transform = _data_transforms_lsun(resize)
            train_transform = train_transform if augment else valid_transform
            data = LSUNClass(root=root + '/cat', transform=train_transform)
            total_examples = len(data)
            train_size = int(0.9 * total_examples)   # use 90% for training
            train_data, valid_data = random_split_dataset(data, [train_size, total_examples - train_size])
        else:
            raise NotImplementedError
    elif dataset.startswith('imagenet'):
        num_classes = 1
        resize = int(dataset.split('_')[1])
        assert root.replace('/', '')[-3:] == dataset.replace('/', '')[-3:], 'the size should match'
        train_transform, valid_transform = _data_transforms_generic(resize)
        train_transform = train_transform if augment else valid_transform
        train_data = LMDBDataset(root=root, name='imagenet-oord', train=True, transform=train_transform)
        valid_data = LMDBDataset(root=root, name='imagenet-oord', train=False, transform=valid_transform)
    elif dataset.startswith('ffhq'):
        num_classes = 1
        resize = 256
        train_transform, valid_transform = _data_transforms_generic(resize)
        train_transform = train_transform if augment else valid_transform
        train_data = LMDBDataset(root=root, name='ffhq', train=True, transform=train_transform)
        valid_data = LMDBDataset(root=root, name='ffhq', train=False, transform=valid_transform)
    else:
        raise NotImplementedError

    train_sampler, valid_sampler = None, None
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_data)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size,
        shuffle=(train_sampler is None) and shuffle_train,
        sampler=train_sampler, pin_memory=True, num_workers=8, drop_last=drop_last_train)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=batch_size,
        shuffle=(valid_sampler is None),
        sampler=valid_sampler, pin_memory=True, num_workers=1, drop_last=False)

    return train_queue, valid_queue, num_classes


def random_split_dataset(dataset, lengths, seed=0):
    """
    Randomly split a dataset into non-overlapping new datasets of given lengths.
    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
    """
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")
    g = torch.Generator()
    g.manual_seed(seed)

    indices = torch.randperm(sum(lengths), generator=g)
    return [torch.utils.data.Subset(dataset, indices[offset - length:offset])
            for offset, length in zip(_accumulate(lengths), lengths)]


def _data_transforms_cifar10():
    """Get data transforms for cifar10."""

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    valid_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    return train_transform, valid_transform


def _data_transforms_mnist(binarize):
    """Get data transforms for mnist."""
    T = [transforms.Pad(padding=2), transforms.ToTensor()]
    if binarize:
        T.append(Binarize())

    train_transform = transforms.Compose(T)
    valid_transform = transforms.Compose(T)

    return train_transform, valid_transform


def _data_transforms_generic(size):
    train_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    valid_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])

    return train_transform, valid_transform


def _data_transforms_celeba64(size):
    train_transform = transforms.Compose([
        CropCelebA64(),
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    valid_transform = transforms.Compose([
        CropCelebA64(),
        transforms.Resize(size),
        transforms.ToTensor(),
    ])

    return train_transform, valid_transform


def _data_transforms_lsun(size):
    train_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.RandomCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    valid_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ])

    return train_transform, valid_transform


