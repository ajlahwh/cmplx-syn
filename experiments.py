"""
All experiments for the paper.
config: the baseline config.
config_ranges: the config items to be varied.
"""
from utils import vary_config
import numpy as np
import pprint
#from birdseye import eye


def co_vary():
    # only for b=9 N=1024; too much time!
    beakers = [9] # , 8, 7, 6, 5, 4, 3, 2, 1]
    seed_num = 10
    config={
        'save_path':'co_vary',
        'isdiscrete': True,
        'burnin_num': 2*10**6,
        'fillin_num': 2*10**6,
        'sample_num': 400,
    }
    configs=[]
    for b in beakers:
        config_ranges = {
            'dim_num': [2**(b+1)],
            'beaker_num': [b],
            'pattern_type': [#'face',
                             'rand'],
            'seed': range(seed_num),
        }
        configs += vary_config(config, config_ranges, mode='combinatorial')

    return configs

def co_vary10():
    # only for b=10 N=2048; too much time!
    beakers = [10]
    seed_num = 30
    config={
        'save_path':'co_vary',
        'isdiscrete': True,
        'burnin_num': 2*10**6,
        'fillin_num': 2*10**6,
        'sample_num': 70, #100,
    }
    configs=[]
    for b in beakers:
        config_ranges = {
            'dim_num': [2**(b+1)],
            'beaker_num': [b],
            'pattern_type': ['face',
                             'rand'],
            'seed': range(10, seed_num),
        }
        configs += vary_config(config, config_ranges, mode='combinatorial')

    return configs


def co_vary_6_facefillin():
    beakers = [6]
    seed_num = 5
    config={
        'save_path':'co_vary_6_facefillin',
        'isdiscrete': True,
        'burnin_num': 2*10**5,
        'fillin_num': 2*10**5,
        'sample_num': 2000,
        'face_fillin_pattern': True,

    }
    configs=[]
    for b in beakers:
        config_ranges = {
            'dim_num': [2**(b+1)],
            'beaker_num': [b],
            'pattern_type': ['face',
                             #'rand'
                             ],
            'seed': range(seed_num),
        }
        configs += vary_config(config, config_ranges, mode='combinatorial')

    return configs


def co_vary_6to1():
    beakers = [6, 5, 4, 3, 2, 1]
    seed_num = 5
    config={
        'save_path':'co_vary',
        'isdiscrete': True,
        'burnin_num': 2*10**5,
        'fillin_num': 2*10**5,
        'sample_num': 2000,
    }
    configs=[]
    for b in beakers:
        config_ranges = {
            'dim_num': [2**(b+1)],
            'beaker_num': [b],
            'pattern_type': ['face',
                             'rand'],
            'seed': range(seed_num),
        }
        configs += vary_config(config, config_ranges, mode='combinatorial')

    return configs

def big_simple():
    seed_num = 15
    config={
        'save_path':'big_simple',
        'isdiscrete': True,
        'beaker_num': 1,
        'burnin_num': 8*10**5,
        'fillin_num': 8*10**5,
        'sample_num': 200,
        'dim_num': 1448,
    }
    config_ranges = {
        'pattern_type': ['face',
                         'rand'],
        'seed': range(seed_num),
    }
    configs = vary_config(config, config_ranges, mode='combinatorial')

    return configs


def big_simple_prob():
    seed_num = 15
    config={
        'save_path':'big_simple_prob',
        'isdiscrete': True,
        'beaker_num': 1,
        'burnin_num': 8*10**5,
        'fillin_num': 8*10**5,
        'sample_num': 200,
        'dim_num': 1448,
    }
    config_ranges = {
        'pattern_type': ['face',
                         'rand'],
        'prob_encoding': [#0.18, 0.16, 0.14, 0.12, 0.1, 0.005
            0.128, 0.005,
            ],
        'seed': range(seed_num),
    }
    configs = vary_config(config, config_ranges, mode='combinatorial')

    return configs

def big_simple_all():
    return big_simple()+big_simple_prob()

def big_as_mel():
    seed_num = 1
    config={
        'save_path':'big_as_mel',
        'isdiscrete': True,
        'burnin_num': 4*10**5,
        'fillin_num': 4*10**5,
        'sample_num': 70,
        'pattern_type': 'rand',
    }
    configs=[]
    for d, b in zip([2862, 2612, 2419, 2263, 2134], [5, 6, 7, 8, 9]):
        config_ranges = {
            'dim_num': [d],
            'beaker_num': [b],
            'prob_encoding': [1, 0.5, 0.2, 0.05, 0.02],
            'seed': range(seed_num),
        }
        configs += vary_config(config, config_ranges, mode='combinatorial')

    return configs


def big_as_mel_match():
    seed_num = 1
    config={
        'save_path':'big_as_mel_match',
        'isdiscrete': True,
        'burnin_num': 4*10**4,
        'fillin_num': 4*10**4,
        'sample_num': 100, #400,
        'pattern_type': 'rand',
    }
    configs=[]
    for d, b in zip(
            [2262, 1600, 1306, 1131, #1012, 923
             ],
            [1, 2, 3, 4, #5, 6
             ]):
        config_ranges = {
            'dim_num': [d],
            'beaker_num': [b],
            'prob_encoding': [1, 0.5, 0.2, 0.1],
            'seed': range(seed_num),
        }
        configs += vary_config(config, config_ranges, mode='combinatorial')

    return configs

def monitor_snr():
    config = {
        'save_path': 'monitor_snr',
        'dim_num': 64,
        'isdiscrete': True,
        'burnin_num': 20000,
        'rpt': 30, # target re-present time
        # 'fillin_num': 200000,
        'sample_num': 100,
    }

    config_ranges = {
        'beaker_num': [7,6,5,1],
        'pattern_type': ['rand'],
        'seed': range(10),
    }

    configs = vary_config(config, config_ranges, mode='combinatorial')

    return configs


def monitor_snr_500_128levels():
    config = {
        'save_path': 'monitor_snr_500_128levels',
        'dim_num': 64,
        'isdiscrete': True,
        'burnin_num': 40000,
        'level_num': 128,
        'rpt': 500, # target re-present time
        # 'fillin_num': 200000,
        'sample_num': 100,
        'monitor_threshold':0.5,
    }

    config_ranges = {
        'beaker_num': [8,7,6,5,4,3,2,1],
        'pattern_type': ['rand'],
        'seed': range(10),
    }

    configs = vary_config(config, config_ranges, mode='combinatorial')

    return configs

def monitor_snr_300():
    config = {
        'save_path': 'monitor_snr_300',
        'dim_num': 64,
        'isdiscrete': True,
        'burnin_num': 40000,
        'rpt': 300, # target re-present time
        # 'fillin_num': 200000,
        'sample_num': 100,
        'monitor_threshold':0.5,
    }

    config_ranges = {
        'beaker_num': [8,7,6,5,4,3,2,1],
        'pattern_type': ['rand'],
        'seed': range(10),
    }

    configs = vary_config(config, config_ranges, mode='combinatorial')

    return configs


def monitor_snr_50():
    config = {
        'save_path': 'monitor_snr_50',
        'dim_num': 64,
        'isdiscrete': True,
        'burnin_num': 40000,
        'rpt': 50, # target re-present time
        # 'fillin_num': 200000,
        'sample_num': 100,
        'monitor_threshold':0.5,
    }

    config_ranges = {
        'beaker_num': [8,7,6,5,1],
        'pattern_type': ['rand'],
        'seed': range(10),
    }

    configs = vary_config(config, config_ranges, mode='combinatorial')

    return configs

def vary_len_b7():
    seed_num = 6
    config={
        'save_path':'vary_len_b7',
        'dim_num': 256,
        'isdiscrete': True,
        'beaker_num': 7,
        'burnin_num': 2*10**5,
        'fillin_num': 2*10**5,
        'sample_num': 2500, # (test_num=3, sample_num=1000, neuron_num=256, neuron_num=256) -> 539M +1498M =2037M
    }

    config_ranges = {
        'beaker_num':[7, 6, 5, 4, 3, 2, 1],
        'pattern_type': ['face', 'rand'],
        'seed': range(seed_num),
    }

    configs = vary_config(config, config_ranges, mode='combinatorial')

    return configs

def fixed_len_b8_level128():
    seed_num = 15
    config={
        'save_path':'fixed_len_b8_level128',
        'dim_num': 512,
        'isdiscrete': True,
        'beaker_num': 8,
        'burnin_num': 8*10**5,
        'fillin_num': 8*10**5,
        'sample_num': 1000, # (test_num=3, sample_num=1000, neuron_num=512, neuron_num=512) -> 577M +5994M =6571M
    }

    config_ranges = {
        'beaker_num':[8, ],
        'pattern_type': ['rand'],
        'seed': range(seed_num),
    }

    configs = vary_config(config, config_ranges, mode='combinatorial')

    return configs

def vary_len_b8():
    seed_num = 15
    config={
        'save_path':'vary_len_b8',
        'dim_num': 512,
        'isdiscrete': True,
        'beaker_num': 8,
        'burnin_num': 8*10**5,
        'fillin_num': 8*10**5,
        'sample_num': 1000, # (test_num=3, sample_num=1000, neuron_num=512, neuron_num=512) -> 577M +5994M =6571M
    }

    config_ranges = {
        'beaker_num':[8, 7, 6, 5, 4, 3, 2, 1],
        'pattern_type': ['face', 'rand'],
        'seed': range(seed_num),
    }

    configs = vary_config(config, config_ranges, mode='combinatorial')

    return configs

def vary_dim():
    seed_num = 3
    config={
        'save_path':'vary_dim',
        'isdiscrete': True,
        'beaker_num': 8,
        'burnin_num': 4*10**5,
        'fillin_num': 2*10**5,
        'sample_num': 2500, # (test_num=3, sample_num=1000, neuron_num=512, neuron_num=512) -> 577M +5994M =6571M
    }

    config_ranges = {
        'dim_num':[256, 128, 64, 32, 16, 8, 4],
        'pattern_type': ['face', 'rand'],
        'seed': range(seed_num),
    }
    configs = vary_config(config, config_ranges, mode='combinatorial')

    return configs

def vary_len():
    return vary_len_b7()+vary_len_b8()


def local_test():

    config={
        'save_path':'local_test',
        'dim_num': 128,
        'isdiscrete': True,
        'burnin_num': 10000,
        'fillin_num': 10000,
        'sample_num': 1000,
    }

    config_ranges = {
        'pattern_type': ['face', 'rand'],
        'beaker_num': [1, 2, 3, 4, 5, 6],
        'seed': [0,],
    }

    configs = vary_config(config, config_ranges, mode='combinatorial')

    return configs


def test_sample_effect():

    config={
        'save_path':'test_sample_effect',
        'dim_num': 128,
        'isdiscrete': True,
        'burnin_num': 1000,
        'fillin_num': 1000,
        'beaker_num': 1,
    }

    config_ranges = {
        'sample_num': [500, 1000, 2000,4000],
        'pattern_type': ['face'],
        'seed': [0],
    }

    configs = vary_config(config, config_ranges, mode='combinatorial')

    return configs


def test_prob():
    config={
        'save_path':'test_prob',
        'dim_num': 128,
        'beaker_num': 1,
        'isdiscrete': True,
        'burnin_num': 1000,
        'fillin_num': 1000,
        'sample_num': 1000,
        'pattern_type': 'rand',
    }

    config_ranges = {
        'prob_encoding': [1, 0.8, 0.6, 0.4, 0.2],
        'seed': [0,],
    }

    configs = vary_config(config, config_ranges, mode='combinatorial')

    return configs


def test_neuron_num():

    config={
        'save_path':'test_neuron_num',
        'isdiscrete': True,
        'burnin_num': 1000,
        'fillin_num': 1000,
        'beaker_num': 1,
    }

    config_ranges = {
        'dim_num': [256, 128, 64, 32, 16],
        'sample_num': [500],
        'pattern_type': ['face'],
        'seed': [0],
    }

    configs = vary_config(config, config_ranges, mode='combinatorial')

    return configs


def test_coding_f():

    config={
        'save_path':'test_coding_f',
        'dim_num': 32,
        'isdiscrete': False,
        'isbounded': False,
        'burnin_num': 20000,
        'fillin_num': 4000,
        'sample_num': 700,
        'beaker_num': 6,
        'pattern_type': 'rand',
    }

    config_ranges = {
        'sparse_coding': [True, ],#False, ],
        'inv_coding_f': [2,2.5,3,4,6,8],#[64, 32, 25, 20, 16, 12,10, 8,6,5,4.5, 4,3.5, 3,2.8,2.6,2.4,2.2,2.1, 2 ], #[32,16,8,4],#
        'seed': [0],
    }
    configs = vary_config(config, config_ranges, mode='combinatorial')

    ref_config_ranges = {
        'sparse_coding': [False],
        'seed': [0],
    }

    ref_configs = vary_config(config, ref_config_ranges, mode='combinatorial')
    return configs + ref_configs


if __name__ == '__main__':
    pprint.pprint(test_coding_f())
