import numpy as np
import torch
import torch.nn as nn
import time
import settings
import roblib
from models import BCSLayer, Memorynet
from datasets import load_aux_patterns, load_traintest_patterns
import experiments
import os.path
import utils
import pprint
import matplotlib.pyplot as plt

def train_experiment(experiment,device='cuda'):
    print('Training {:s} experiment'.format(experiment))
    configs = getattr(experiments, experiment)()
    for config in configs:
        train(config,device=device)
        if device == 'cuda':
            torch.cuda.empty_cache()


def monitor_experiment(experiment,device='cuda'):
    print('Training & mornitoring {:s} experiment'.format(experiment))
    configs = getattr(experiments, experiment)()
    for config in configs:
        monitor(config,device=device)
        if device == 'cuda':
            torch.cuda.empty_cache()


def get_test_time_point(max_time_point, inval_type='log'):
    L = []
    if inval_type == 'log':
        st = 0
        i = 0
        while(True):
            L.append(np.arange(st, st + 10 * 2 ** i, 2 ** i))
            st += 10 * 2 ** i
            i += 1
            if st>max_time_point:
                break
        L = np.concatenate(L, 0)
        L=L[L<max_time_point]
    elif inval_type == 'lin':
        L = np.arange(max_time_point)
    neg_times= -L[::-1] - 1
    pos_times = L
    return neg_times, pos_times
    # array([...,
    # -311287, -294903, -278519, -262135, -245751, -229367, -212983,...
    # -3,      -2,      -1,]       [0,       1,       2,       3,
    # 229366,  245750,  262134,  278518,  294902,  311286,
    # ...])

def train(config, device='cuda'):
    plot_train_process = True
    path=settings.MODELPATH / config['save_path']
    os.makedirs(path, exist_ok=True)
    filename = path / 'simulation.bk'
    if os.path.exists(filename):
        return

    utils.save_config(config, config['save_path'])
    pprint.pprint(config)
    print('Simulating ',filename)
    burnin_num = 196608 if 'burnin_num' not in config else config['burnin_num']
    sample_num = 2000 if 'sample_num' not in config else config['sample_num']
    fillin_num = 196608 if 'fillin_num' not in config else config['fillin_num']
    pattern_type = config['pattern_type']
    dim_num = config['dim_num']
    np.random.seed(config['seed'])
    if 'sparse_coding' in config and config['sparse_coding']:
        sparse_coding = config['sparse_coding']
        coding_f = 1/config['inv_coding_f']
    else:
        sparse_coding = False
        coding_f = 0.5
    traintest_features, test_type = load_traintest_patterns(sample_num, dim_num, pattern_type=pattern_type, sparse_coding=sparse_coding, coding_f=coding_f)
    # [3, sample_num, feature_num]
    burnin_features = load_aux_patterns(burnin_num, dim_num, aug_pattern_type=pattern_type, sparse_coding=sparse_coding, coding_f=coding_f)
    fillin_features = load_aux_patterns(fillin_num, dim_num, aug_pattern_type=pattern_type, sparse_coding=sparse_coding, coding_f=coding_f)
    memorynet = Memorynet(device=device, **config)
    print('Memorynet created')
    neg_times, pos_times = get_test_time_point(fillin_num, 'log')
    pt = 2  # 2 # better use protocol 2 in order to compute the task performance; or use 3

    save_weight_num = 0
    start = time.time()
    num_per_block = burnin_num #//60
    cur_num = 0
    while cur_num<burnin_num:
        next_num=min(cur_num+num_per_block, burnin_num)
        print(f'Training burnin samples from {cur_num} to {next_num}')
        all_weight = memorynet.train_all_neurons(
            burnin_features[cur_num:next_num], save_weight=save_weight_num, burnin_stage=True)
        # memorynet.show_neurons_coef_distribution()
        cur_num = next_num
        # if cur_num>10*num_per_block:
        #     syss.exit()

    print(f'Training burnin done. second passed', time.time() - start)
    # syss.exit()
    r_signal_famil, io_signal_famil, times_famil, \
    r_signal_novel, io_signal_novel, times_novel = memorynet.build_dataset_protocol_during_training(
        traintest_features, fillin_features, neg_times, pos_times, protocol=pt)
    print(f'Training sample+fillin done. second passed', time.time() - start)

    roblib.dump({'r_signal_famil': r_signal_famil,
                 'io_signal_famil': io_signal_famil,
                 'times_famil': times_famil,
                 'r_signal_novel': r_signal_novel,
                 'io_signal_novel': io_signal_novel,
                 'times_novel': times_novel,
                 'test_type': test_type,
                 }, filename)

    print('second passed', time.time() - start)


def monitor(config, device='cuda'):
    path=settings.MODELPATH / config['save_path']
    os.makedirs(path, exist_ok=True)
    filename = path / 'monitor_simulation.bk'
    # if os.path.exists(filename):
    #     return

    utils.save_config(config, config['save_path'])
    pprint.pprint(config)
    print('Simulating & monitoring',filename)
    burnin_num = 4000 if 'burnin_num' not in config else config['burnin_num']
    sample_num = 500 if 'sample_num' not in config else config['sample_num']
    fillin_num = 4000 if 'fillin_num' not in config else config['fillin_num']
    pattern_type = config['pattern_type']
    dim_num = config['dim_num']
    np.random.seed(config['seed'])
    if 'sparse_coding' in config and config['sparse_coding']:
        sparse_coding = config['sparse_coding']
        coding_f = 1/config['inv_coding_f']
    else:
        sparse_coding = False
        coding_f = 0.5
    traintest_features, test_type = load_traintest_patterns(sample_num, dim_num, pattern_type=pattern_type, sparse_coding=sparse_coding, coding_f=coding_f)
    # [3, sample_num, feature_num]
    burnin_features = load_aux_patterns(burnin_num, dim_num, aug_pattern_type=pattern_type, sparse_coding=sparse_coding, coding_f=coding_f)

    memorynet = Memorynet(device=device, **config)
    print('Memorynet created')
    save_weight_num = 0
    start = time.time()
    num_per_block = burnin_num #//60
    cur_num = 0
    while cur_num<burnin_num:
        next_num=min(cur_num+num_per_block, burnin_num)
        print(f'Training burnin samples from {cur_num} to {next_num}')
        all_weight = memorynet.train_all_neurons(
            burnin_features[cur_num:next_num], save_weight=save_weight_num, burnin_stage=True)
        # memorynet.show_neurons_coef_distribution()
        cur_num = next_num
        # if cur_num>10*num_per_block:
        #     syss.exit()

    print(f'Training burnin done. second passed', time.time() - start)
    r_signal_all = []
    io_signal_all = []
    present_time_all = []
    monitor_sample_num = sample_num # 100
    for idx in range(monitor_sample_num):
        print(idx,end=',')
        fillin_features = load_aux_patterns(fillin_num, dim_num, aug_pattern_type=pattern_type,
                                            sparse_coding=sparse_coding, coding_f=coding_f,verbose=False)
        r_signal, io_signal, present_time = memorynet.monitor_snr_during_training(
            traintest_features[0, idx:idx+1], fillin_features,
            monitored_signal_thre=0.3)
        r_signal_all.append(r_signal)
        io_signal_all.append(io_signal)
        present_time_all.append(present_time)
    print(f'\nTraining sample+fillin done. second passed', time.time() - start)

    d = {'r_signal_famil': r_signal_all,
                 'io_signal_famil': io_signal_all,
                 'present_time': present_time_all,
                 'test_type': test_type,
                 }
    roblib.dump(d, filename)

    print('second passed', time.time() - start)

    # for idx in range(10):
    #     plt.subplot(3,1,1)
    #     plt.plot(d['io_signal_famil'][idx], alpha=0.5)
    #     plt.vlines(d['present_time'][idx],0,1, 'r')
    #     plt.subplot(3,1,2)
    #     plt.plot(d['r_signal_famil'][idx], alpha=0.5)
    #     plt.vlines(d['present_time'][idx],0,1, 'r')
    #     plt.subplot(3,1,3)
    #     plt.plot(np.diff(d['present_time'][idx]))
    #     plt.show()