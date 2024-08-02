#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=500, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_bs', type=int, default=5, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--dataset', type=str, default='cifar10', help="name of dataset")
    parser.add_argument('--all_clients', action='store_true', help='aggregation over all clients')

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs1', type=int, default=10,help='the number of local epochs')
    # parser.add_argument('--seed', type=int, default=1707208909)
    parser.add_argument('--start_rate', type=float, default=0.0)
    parser.add_argument('--grow_epochs', type=int, default=10)
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--load_from_points', action='store_true', help='whether to continue training from checkpoint')
    if parser.parse_args().load_from_points:
        parser.add_argument('--point', type=str, default='399', help='checkpoint to load')
        parser.add_argument('--further_epochs', type=int, default=150, help='further training epochs')

    parser.add_argument('--data_beta', type=float, default=0.5, help="Dirichlet")

    parser.add_argument('--patience', type=int, default=10, help="number of how many rounds to wait for the best acc")
    parser.add_argument('--reverse', action='store_true', help='whether reverse or not')
    parser.add_argument('--easy_threshold', type=float, default=0.5, help="number of rounds for training easy data")
    parser.add_argument('--midle_threshold', type=float, default=0.8, help="number of rounds for training midle data")

    args = parser.parse_args()
    return args
