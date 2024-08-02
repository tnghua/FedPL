#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn

def count(user_datanumber_list):
    data_num_each = []
    all_data_num = 0
    for idx in range(len(user_datanumber_list)):
        idxs = user_datanumber_list[idx]
        data_num_each.append(idxs)
        all_data_num += idxs
    return all_data_num, data_num_each


def FedAvg(w,user_datanumber_list):
    list_p = {}
    allnum, eachnum = count(user_datanumber_list)
    for m in range(len(user_datanumber_list)):
        list_p[m] = eachnum[m] / allnum
    w_avg = copy.deepcopy(w[0])
    w_avg = dict.fromkeys(w_avg, 0)

    for k in w_avg.keys():
        for i in range(len(w)):

            w_avg[k] += torch.mul(w[i][k], list_p[i])

    return w_avg

