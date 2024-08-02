
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import os
import sympy
import scipy.integrate as spi
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
from utils import mydata
from utils.sampling import cifar_iid,fashion_mnist_iid, dirichlet_split_noniid
from utils.options import args_parser
from models.Updatenew import LocalUpdate
from models.Nets import VGG9, VGG11, VGG16, ResNet18, ResNet34, ResNet50, CNNCifar, CNNMnist, CNNSVNH
from models.Fed import FedAvg
from models.test import test_img
import time
import datetime
import random

args = args_parser()

if args.load_from_points:
    checkpoint = torch.load('./checkpoints/{}_{}_{}_{}/checkpoint_{}.pth'.format(args.data_beta, args.local_bs, args.momentum, args.point, args.seed))
    rand_seed = args.seed
else:
    rand_seed = int(time.time())

def set_rand_seed(seed=rand_seed):
    print("Random Seed: ", seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    # parse args
    set_rand_seed()
    print('========= easy_threshold:{}; middle_threshold:{} =========\n'.format(args.easy_threshold,
                                                                                args.midle_threshold))
    '''
    data preparation
    '''
    trans_cifar = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    trans_mnist = transforms.Compose([transforms.ToTensor(),transforms.Normalize(0.286, 0.353)])
    trans_tin = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    trans_svhn = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trans_cinic = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.47889522, 0.47227842, 0.43047404), (0.24205776, 0.23828046, 0.25874835))])

    if args.dataset == 'cifar10':

        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users, rand_seed)
        else:
            labels = np.array(dataset_train.targets)
            dict_users = dirichlet_split_noniid(labels, alpha=args.data_beta, n_clients=args.num_users)

    elif args.dataset == 'cifar100':
        dataset_train = datasets.CIFAR100('../data/cifar100', train=True, download=True,
                                          transform=trans_cifar)
        dataset_test = datasets.CIFAR100('../data/cifar100', train=False, download=True,
                                         transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users, rand_seed)
        else:
            labels = np.array(dataset_train.targets)
            dict_users = dirichlet_split_noniid(labels, alpha=args.data_beta, n_clients=args.num_users)

    elif args.dataset == 'fashion-mnist':
        dataset_train = datasets.FashionMNIST('./data/fashion-mnist/', train=True,
                                              download=True, transform=trans_mnist)
        dataset_test = datasets.FashionMNIST('./data/fashion-mnist/', train=False,
                                             download=True, transform=trans_mnist)
        if args.iid:
            dict_users = fashion_mnist_iid(dataset_train, args.num_users, rand_seed)
        else:
            labels = np.array(dataset_train.targets)
            dict_users = dirichlet_split_noniid(labels, alpha=args.data_beta, n_clients=args.num_users)

    elif args.dataset == 'svhn':
        dataset_train = datasets.SVHN('../data/svhn', split='train', download=True, transform=trans_svhn)
        dataset_test = datasets.SVHN('../data/svhn', split='test', download=True, transform=trans_svhn)
        if args.iid:
            dict_users = fashion_mnist_iid(dataset_train, args.num_users, rand_seed)
        else:
            dataset_train.targets = dataset_train.labels
            dataset_test.targets = dataset_test.labels
            labels = np.array(dataset_train.targets)
            dict_users = dirichlet_split_noniid(labels, alpha=args.data_beta, n_clients=args.num_users)

    elif args.dataset == 'cinic10':
        dataset_train = datasets.ImageFolder('../data/CINIC-10/train', transform=trans_cinic)
        dataset_test = datasets.ImageFolder('../data/CINIC-10/test', transform=trans_cinic)
        if args.iid:
            dict_users = fashion_mnist_iid(dataset_train, args.num_users, rand_seed)
        else:
            labels = np.array(dataset_train.targets)
            dict_users = dirichlet_split_noniid(labels, alpha=args.data_beta, n_clients=args.num_users)

    elif args.model == 'resnet18' and args.dataset == 'cifar100':
        net_glob = ResNet18(args)
    elif args.model == 'cnn' and args.dataset == 'fashion-mnist':
        net_glob = CNNMnist(args)
    elif args.model == 'cnn' and args.dataset == 'cifar10':
        net_glob = CNNCifar(args)
    elif args.model == 'cnn' and args.dataset == 'svhn':
        net_glob = CNNSVNH(args)
    elif args.model == 'cnn' and args.dataset == 'cinic10':
        net_glob = CNNCifar(args)

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    time_begin = time.time()
    avgloss_locals_all = []
    sd_locals_all = []
    find_min_all = []
    find_max_all = []
    client_prob_first = []
    client_thr_data_num = []
    easy_clients=[]
    midle_clients = []
    hard_clients = []
    midround_log = 0
    easyround_log = 0
    stage_counter = 0
    best_loss = np.inf
    prob_flag = 0
    stage = 'easy'
    if not args.load_from_points:
        iter = 0
        total_epochs = args.epochs
    else:
        iter = checkpoint['iter']
        stage = checkpoint['stage']
        net_glob.load_state_dict(checkpoint['state_dict'])
        best_loss = checkpoint['best_loss']
        stage_counter = checkpoint['stage_counter']
        total_epochs = args.further_epochs
        prob_flag = checkpoint['prob_flag']

    net_glob.train()
    # copy weights
    w_glob = net_glob.state_dict()

    for iter in range(iter, total_epochs):
        if stage == 'easy':
            if prob_flag == 0:
                prob_flag = 1
                P_total = 0
                for idx in range(args.num_users):
                    torch.cuda.empty_cache()
                    net_loss = copy.deepcopy(net_glob)
                    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], num_epochs=args.epochs1,
                                        device_name=args.device, net_fix=net_loss)
                    avgloss_locals,sd_locals= local.loss_test(net_threshold=net_loss)
                    avgloss_locals_all.append(avgloss_locals)
                    sd_locals_all.append(sd_locals)
                    del net_loss
                    print("idx_users", idx)
                for index, item in enumerate(avgloss_locals_all):
                    find_min_all.append(item - 3 * sd_locals_all[index])
                    find_max_all.append(item + 3 * sd_locals_all[index])
                min_loss_all = max(min(find_min_all), 0)
                max_loss_all = max(find_max_all)
                easy_thr_client = 0
                midle_thr_client = min_loss_all + (max_loss_all-min_loss_all) * args.easy_threshold
                for idx in range(args.num_users):
                    x, u, sita = sympy.symbols('x u sita')
                    u = avgloss_locals_all[idx]
                    sita = sd_locals_all[idx]
                    up = midle_thr_client
                    down = 0
                    fx = 1 / (sita * sympy.sqrt(2 * sympy.pi)) * sympy.exp(-((x - u) * (x - u)) / (2 * sita * sita))
                    def integrand(x, u, sita):
                        fx = 1 / (sita * sympy.sqrt(2 * sympy.pi)) * sympy.exp(-((x - u) * (x - u)) / (2 * sita * sita))
                        return fx
                    result, error = spi.quad(integrand, down, up, args=(u, sita))
                    P=result
                    P_total= P+P_total
                    client_prob_first.append(P)

                for data_idx in range(args.num_users):
                    client_thr_data_num.append(len(dict_users[data_idx])*client_prob_first[data_idx])
                total_thr_data_num = sum(client_thr_data_num)
                client_prob=[item / total_thr_data_num for item in client_thr_data_num]
                i=0
                for item in client_prob:
                    i+=1
                    Name = 'clients_{}_C[{:.2f}]]'.format(args.dataset, args.frac)
                    file_name = Name
                    file = open(file_name + ".txt", 'a', encoding='utf-8')
                    file.write('{}\t{:.3f}\n'.format(i, item))

            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, p=client_prob, replace=False)

        if stage == 'middle':
            if prob_flag == 0:
                prob_flag = 1
                avgloss_locals_all.clear()
                sd_locals_all.clear()
                find_min_all.clear()
                find_max_all.clear()
                client_prob_first.clear()
                client_prob.clear()
                client_thr_data_num.clear()
                P_total = 0
                for idx in range(args.num_users):
                    torch.cuda.empty_cache()
                    net_loss = copy.deepcopy(net_glob)
                    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], num_epochs=args.epochs1,
                                        device_name=args.device, net_fix=net_loss)
                    avgloss_locals,sd_locals= local.loss_test(net_threshold=net_loss)
                    avgloss_locals_all.append(avgloss_locals)
                    sd_locals_all.append(sd_locals)
                    del net_loss
                    print("idx_users", idx)

                for index, item in enumerate(avgloss_locals_all):
                    find_min_all.append(item - 3 * sd_locals_all[index])
                    find_max_all.append(item + 3 * sd_locals_all[index])
                print("find_min_all", find_min_all)
                print("find_max_all", find_max_all)
                min_loss_all = max(min(find_min_all), 0)
                max_loss_all = max(find_max_all)
                easy_thr_client = min_loss_all + (max_loss_all - min_loss_all) * args.easy_threshold
                midle_thr_client = min_loss_all + (max_loss_all-min_loss_all) * args.midle_threshold
                for idx in range(args.num_users):
                    x, u, sita = sympy.symbols('x u sita')
                    u = avgloss_locals_all[idx]
                    sita = sd_locals_all[idx]
                    up = midle_thr_client
                    down = 0
                    fx = 1 / (sita * sympy.sqrt(2 * sympy.pi)) * sympy.exp(-((x - u) * (x - u)) / (2 * sita * sita))
                    def integrand(x, u, sita):
                        fx = 1 / (sita * sympy.sqrt(2 * sympy.pi)) * sympy.exp(-((x - u) * (x - u)) / (2 * sita * sita))
                        return fx
                    result, error = spi.quad(integrand, down, up, args=(u, sita))
                    P=result
                    P_total= P+P_total
                    client_prob_first.append(P)

                for data_idx in range(args.num_users):
                    client_thr_data_num.append(len(dict_users[data_idx]) * client_prob_first[data_idx])
                total_thr_data_num = sum(client_thr_data_num)
                client_prob = [item / total_thr_data_num for item in client_thr_data_num]
                i = 0
                for item in client_prob:
                    i += 1
                    print(i, item)
                    Name = 'clients_{}_C[{:.2f}]]'.format(args.dataset, args.frac)
                    file_name = Name
                    file = open(file_name + ".txt", 'a', encoding='utf-8')
                    file.write('{}\t{:.3f}\n'.format(i, item))
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, p=client_prob, replace=False)

        if stage == 'hard':
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        loss_locals = []
        if not args.all_clients:
            w_locals = []

        avgloss_locals = []
        sd_locals = []
        find_min = []
        find_max = []
        users_batch_loss = []  # save all users batch loss

        for idx in idxs_users:
            torch.cuda.empty_cache()
            net_loss = copy.deepcopy(net_glob)
            net_client_fix = copy.deepcopy(net_glob)
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], num_epochs=args.epochs1,
                                device_name=args.device, net_fix=net_client_fix)
            avglocal, sdlocal = local.loss_test(net_threshold=net_loss)
            users_batch_loss.append(local.get_batch_loss())
            avgloss_locals.append(avglocal)
            sd_locals.append(sdlocal)
            del net_loss, net_client_fix
            print("idx_users", idx)

        for index, item in enumerate(avgloss_locals):
            find_min.append(item - 3 * sd_locals[index])
            find_max.append(item + 3 * sd_locals[index])
        min_loss = max(min(find_min), 0)
        max_loss = max(find_max)
        easyloss = min_loss + (max_loss - min_loss) * args.easy_threshold
        midleloss = min_loss + (max_loss-min_loss)* args.midle_threshold

        if stage == 'easy':
            threshold = easyloss
        elif stage == 'middle':
            threshold = midleloss
        else:
            threshold = 10000

        print("threshold", threshold)

        idx_loss = 0
        idx_valid = []
        flag_none = 0
        user_loss_list = []
        user_acc_list = []
        user_datanumber_list = []
        for idx in idxs_users:
            torch.cuda.empty_cache()
            net_client = copy.deepcopy(net_glob)
            net_client_fix = copy.deepcopy(net_glob)
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], num_epochs=args.epochs1,
                                device_name=args.device, net_fix=net_client_fix)  # current a new local
            local.set_batch_loss(users_batch_loss[idx_loss])
            idx_loss += 1
            lossthreshold = threshold

            w, loss, log_dict = local._train(netName=net_client, threshold=lossthreshold) #  receive log_dict
            user_loss_list.append(log_dict['train_loss'])
            user_acc_list.append(log_dict['train_acc'])
            if w == None and loss == 0:
                idxs_users = np.delete(idxs_users, np.where(idxs_users == idx))
                flag_none = 1
            else:
                w_locals.append(copy.deepcopy(w))
                loss_locals.append(loss)
                user_datanumber_list.append(log_dict['datanumber'])
            del net_client, net_client_fix

        # update global weights
        if flag_none == 0:
            w_glob = FedAvg(w_locals, user_datanumber_list)  # 传入新客户端列表
        else:
            flag_none = 0
            if len(w_locals) == 0:
                print('No valid clients!')
                continue
            else:
                w_glob = FedAvg(w_locals, user_datanumber_list)

        net_glob.load_state_dict(w_glob)
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)  # # plot loss curve

        net_glob.eval()
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        print("Testing accuracy: {:.2f}".format(acc_test))

        user_avg_loss = sum(user_loss_list) / len(user_loss_list)
        user_avg_acc = sum(user_acc_list) / len(user_acc_list)

        if user_avg_loss < best_loss:
            best_loss = user_avg_loss
            stage_counter = 0
        else:
            stage_counter += 1

        if stage_counter >= args.patience and stage == 'easy':
            stage = 'middle'
            easyround_log = iter
            stage_counter = 0
            prob_flag = 0
        elif stage_counter >= args.patience and stage == 'middle':
            stage = 'hard'
            midround_log = iter
            stage_counter = 0
            prob_flag = 0

        if args.iid:
            if not os.path.exists('./iid_log/'):
                os.makedirs('./iid_log/')
            Name = 'newAgg_iid_{}_{}_[beta:{}]_[epochs1:{}]_[total_clients:{}]_[aggre_clients:{}]_[thr:{},{}]'.format(args.dataset, args.model, args.data_beta, args.epochs1, args.num_users, str(args.frac*args.num_users), args.easy_threshold, args.midle_threshold)
            file_path = './iid_log/' + Name + '.txt'
            file = open(file_path, 'a', encoding='utf-8')
            file.write('{}\t{}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n'.format(str(stage), iter, user_avg_loss, user_avg_acc, loss_test, acc_test))

        elif not args.iid:
            if not os.path.exists('./noniid_log/'):
                os.makedirs('./noniid_log/')
            Name = 'lin_new_Agg_noniid_{}_{}_{}_[bas:{}]_[pa:{}]_[beta:{}]_[seed:{}]_[bs:{}]_[momentum:{}]_[epochs1:{}]_[total_clients:{}]_[aggre_clients:{}]_[thr:{},{}]'.format(args.dataset, args.model, args.lr, args.local_bs,args.patience, args.lr,args.data_beta, rand_seed, args.local_bs, args.momentum, args.epochs1, args.num_users, str(args.frac*args.num_users), args.easy_threshold, args.midle_threshold,)
            file_path = './noniid_log/' + Name + '.txt'
            file = open(file_path, 'a', encoding='utf-8')
            file.write('{}\t{}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n'.format(str(stage), iter, user_avg_loss, user_avg_acc, loss_test, acc_test))
            if (iter+1) % 20 == 0:
                if not os.path.exists('./checkpoints/{}_{}_{}_{}/'.format(args.data_beta, args.local_bs, args.momentum, rand_seed)):
                    os.makedirs('./checkpoints/{}_{}_{}_{}/'.format(args.data_beta, args.local_bs, args.momentum, rand_seed))
                training_state = {
                    'iter': iter,
                    'state_dict': net_glob.state_dict(),
                    'stage': stage,
                    'stage_counter': stage_counter,
                    'best_loss': best_loss,
                    'prob_flag': prob_flag,
                    'seed': rand_seed,
                }
                torch.save(training_state, './checkpoints/{}_{}_{}_{}/checkpoint_{}.pth'.format(args.data_beta, args.local_bs, args.momentum,rand_seed, iter))

    time_end = time.time()
    total_time = time_end - time_begin
    print('time:', total_time)
    file = open(file_path, 'a', encoding='utf-8')
    file.write('{:.3f}\n'.format(total_time))
    finish_datetime = datetime.datetime.now()
    file.write('{}\n'.format(finish_datetime))
    file.close()

