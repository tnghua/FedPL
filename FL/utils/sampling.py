#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms
from models.Updatenew import DatasetSplit
def mnist_iid(dataset, num_users,seed):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    np.random.seed(seed)
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users,seed):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    np.random.seed(seed)
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.targets.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def cifar_iid(dataset, num_users,seed):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    np.random.seed(seed)
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def fashion_mnist_iid(dataset, num_users,seed):
    np.random.seed(seed)
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])

    for i in range(num_users):
        dict_users[i] = np.array(list(dict_users[i])).tolist()
    return dict_users


def dirichlet_split_noniid(train_labels, alpha, n_clients):
    min_size_train = 0
    min_require_size = 10
    while min_size_train < min_require_size:
        n_classes = train_labels.max()+1
        label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)
        class_idcs = [np.argwhere(train_labels == y).flatten()
                    for y in range(n_classes)]
        client_idcs = [[] for _ in range(n_clients)]
        for k_idcs, fracs in zip(class_idcs, label_distribution):
            for i, idcs in enumerate(np.split(k_idcs,
                                            (np.cumsum(fracs)[:-1]*len(k_idcs)).
                                            astype(int))):
                client_idcs[i] += [idcs]

        client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
        min_size_train = min([len(idx_j) for idx_j in client_idcs])

    return client_idcs


if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
