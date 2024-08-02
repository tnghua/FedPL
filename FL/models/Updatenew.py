#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import math
import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms
import threading
# 新加
import os
import time
import logging
from .cifar10 import get_cifar10_dataset

import pandas as pd
from scipy import stats
from .resnet import *
import torch.nn as nn


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

class LocalUpdate(object):
    def __init__(self, args, num_epochs, device_name, net_fix, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.device = torch.device(device_name if torch.cuda.is_available() else 'cpu')
        self.ldr_train = DatasetSplit(dataset, idxs)
        self.ldr_loss_test = DataLoader(DatasetSplit(dataset, idxs), shuffle=True)
        self.net = net_fix
        self.net.to(self.device)

        self.epoch = 0
        self.epochs = num_epochs
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.data_size = len(self.ldr_train)
        self.batch_size = args.local_bs
        self.start_rate = args.start_rate
        self.grow_epochs = args.grow_epochs

        self.batch_loss = []
        self.reverse = args.reverse

    def get_batch_loss(self):
        return self.batch_loss

    def set_batch_loss(self, my_batch_loss):
        self.batch_loss = my_batch_loss

    def data_curriculum(self):
        if (self.epoch >= self.epochs):
            self.epoch = 0
        self.epoch += 1
        data_rate = min(1.0, self._subset_grow())  # Current proportion of sampled data.
        data_loss = self.batch_loss
        datanumber = 0

        for data_loss_i in data_loss:
            if data_loss_i <= self.lossthreshold:
                datanumber += 1
        if datanumber==0:
            print("datanumber", datanumber)
            return 0, datanumber
        print("客户端数据", len(self.ldr_train))
        print("datanumber", datanumber)

        data_size = int(datanumber * data_rate)
        data_loss = torch.Tensor(data_loss)
        data_indices = torch.argsort(data_loss)[
                       :data_size]  # Sort data according to the loss value and sample the easist data.
        dataset = Subset(self.ldr_train, data_indices)

        try:
            return DataLoader(dataset, self.batch_size, shuffle=True), datanumber
        except:
            return 0, datanumber

    def loss_curriculum(self, criterion, outputs, labels):
        return torch.mean(criterion(outputs, labels))

    def _subset_grow(self):
        return self.start_rate + (1.0 - self.start_rate) / self.grow_epochs * self.epoch

    def _loss_measure(self):
        return torch.cat([self.criterion(
            self.net(data[0].to(self.device)), data[1].to(self.device)).detach()
                          for data in DataLoader(self.ldr_train, self.batch_size)])

    def loss_test(self, net_threshold):
        net_threshold.train()
        # train and update
        optimizer = torch.optim.SGD(net_threshold.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        net_threshold.to(self.device)
        for step, (images, labels) in enumerate(self.ldr_loss_test):
            images, labels = images.to(self.device), labels.to(self.device)
            net_threshold.zero_grad()
            log_probs = net_threshold(images)
            loss = self.loss_func(log_probs, labels)
            loss.backward()
            self.batch_loss.append(loss.item())
        avg = np.mean(self.batch_loss)
        sd = np.std(self.batch_loss)
        return avg, sd

    def _train(self, netName, threshold):
        best_acc = 0.0
        epoch_loss = []
        net = netName.to(self.device)
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=5e-4)
        self.lossthreshold =threshold
        net.train()
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-6)
        softmax = nn.Softmax(dim=1)
        log_datanumber = 0
        acc_list = []
        for epoch in range(self.epochs):
            batch_loss = []
            t = time.time()
            total = 0
            correct = 0
            train_loss = 0.0
            train_data = 0
            train_acc = 0.0
            worker_time = 0.0
            loader, datanumber = self.data_curriculum() #接收loader and datanumber
            log_datanumber = datanumber
            if loader==0:
                log_dict = {'datanumber': datanumber,
                            'train_data': 0,
                            'train_acc': 0.0,
                            'train_loss': 0.0
                            }
                return None, 0, log_dict
           
            for step, (images, labels) in enumerate(loader):
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = net(images)

                loss = self.loss_curriculum(self.criterion, outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                _, predicted = outputs.max(dim=1)
                correct += predicted.eq(labels).sum().item()
                total += labels.shape[0]
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

            lr_scheduler.step()
            print('[%3d]  Train data = %6d  Train Acc = %.4f  Loss = %.4f  Time = %.2f' % (
            epoch + 1, total, correct / total, train_loss / (step + 1), time.time() - t))
            acc_list.append(correct / total)
      
        log_dict = {'datanumber': log_datanumber,
                    'train_data':len(self.ldr_train),
                    'train_acc':sum(acc_list) / len(acc_list),
                    'train_loss':sum(epoch_loss) / len(epoch_loss)
                    }
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), log_dict  # 以字典的方式传递log


data_dict = {
    'cifar10': get_cifar10_dataset,
}

def get_dataset_with_noise(data_dir, data_name):
    if 'noise' in data_name:
        try:
            parts = data_name.split('-')
            data_name = parts[0]
            noise_ratio = float(parts[-1])
        except:
            assert False, 'Assert Error: data_name shoule be [dataset]-noise-[ratio]'
        assert noise_ratio >= 0.0 and noise_ratio <= 1.0, \
            'Assert Error: noise ratio should be in range of [0.0, 1.0]'
    else:
        noise_ratio = 0.0

    assert data_name in data_dict, \
        'Assert Error: data_name should be in ' + str(list(data_dict.keys()))
    return data_dict[data_name](data_dir, noise_ratio=noise_ratio)


def get_net(net_name):
    net_dict = {
        'resnet': ResNet18,
        # TODO: more version of the nets above
    }

    return net_dict[net_name](num_classes=10)

def get_logger(log_file, log_name=None):
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)

    log_format = logging.Formatter(
        fmt='%(asctime)s\t%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    fh = logging.FileHandler(log_file)
    fh.setFormatter(log_format)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(log_format)
    logger.addHandler(ch)

    return logger

def set_random(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
