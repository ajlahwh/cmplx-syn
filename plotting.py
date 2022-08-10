"""
Plotting all experiments for the paper.
"""

import numpy as np
import roblib
import settings
from pathlib import Path
import pandas as pd
from utils import load_config
from plot_maker import plot_module

def naming_unfolding(str_template, unfolding_specs):
    cur_strings = [str_template]
    for idx, spec in enumerate(unfolding_specs):
        if len(spec) == 2:
            re_match, var_range = spec
            colors = [None]*len(var_range)
        else:
            assert idx+1==len(unfolding_specs) # only the last variable can specify colors
            re_match, var_range, colors = spec
        assert re_match[-2:]=='-?'
        re_replace = re_match[:-1]
        new_strings = []
        for cur_str in cur_strings:
            for var, color in zip(var_range, colors):
                s = cur_str.replace(re_match, re_replace+str(var))
                new_strings.append(s if color is None else (s, var, color))
        cur_strings = new_strings
    print(cur_strings)
    return cur_strings

def model_iter_beakers(str_template, pattern_types, b_max):
    compared_models = []
    for f in pattern_types:
        for b in range(1,b_max+1):
            compared_models.append([str_template.replace('beak-?', f'beak-{b}').replace('patt-?', f'patt-{f}'),
                b, 1-b/b_max]) # path, id, color
    return compared_models

def local_test(name_filter=''):
    plot_module(
        'local_test',
        naming_unfolding('local_test/model_patt-?_beak-?_seed',
                         [('patt-?', ['face', 'rand']),
                          ('beak-?', np.arange(1,7), 1-np.arange(1,7)/6)]),
        {'legendtitle': 'Synaptic \ncomplexity'}
    )

def vary_len_b7(name_filter=''):
    plot_module(
        'vary_len_b7',
        naming_unfolding('vary_len_b7/model_beak-?_patt-?_seed',
                         [('patt-?', ['face', 'rand']),
                          ('beak-?', np.arange(1, 8), 1 - np.arange(1, 8) / 7)]),
        {'legendtitle': 'Synaptic \ncomplexity'}
    )


def vary_len_b8(name_filter=''):
    plot_module(
        'vary_len_b8',
        naming_unfolding('vary_len_b8/model_beak-?_patt-?_seed',
                         [('patt-?', ['face', 'rand']),
                          ('beak-?', np.arange(1, 9), 1 - np.arange(1, 9) / 8)]),
        {'legendtitle': 'Synaptic \ncomplexity','tstar_type': 'linear',
         # 'additional': ['SNRratio'],
         'no_regr_models': {
             ('face', 'same'): [7,8],
             ('face', 'noisy'): [6,7, 8],
             ('rand', 'same'): [7, 8],
         }
         },
    )

def co_vary_6_facefillin(name_filter=''):
    plot_module(
        'face_fillin',
        [
            ('co_vary/model_dim_-128_beak-6_patt-face_seed', 'recent random', 1 / 7),
            ('co_vary_6_facefillin/model_dim_-128_beak-6_patt-face_seed', 'recent face', 6 / 7),
         ],
        {'legendtitle': '',#'tstar_type': 'log',

         # 'no_regr_models': {
         #     ('face', 'same'): [7,8],
         #     ('face', 'noisy'): [6,7, 8],
         #     ('rand', 'same'): [7, 8],
         # }
         },
    )

def vary_dim(name_filter=''):
    plot_module(
        'vary_dim',
        [
            ('vary_dim/model_dim_-16_patt-face_seed', 16, 6/6),
            ('vary_dim/model_dim_-32_patt-face_seed', 32, 5/6),
            ('vary_dim/model_dim_-64_patt-face_seed', 64, 4/6),
            ('vary_dim/model_dim_-128_patt-face_seed', 128, 3/6),
            ('vary_dim/model_dim_-256_patt-face_seed', 256, 2/6),
            ('vary_len_b8/model_beak-8_patt-face_seed',512,1/6),
            ('vary_dim/model_dim_-16_patt-rand_seed', 16, 6/6),
            ('vary_dim/model_dim_-32_patt-rand_seed', 32, 5/6),
            ('vary_dim/model_dim_-64_patt-rand_seed', 64, 4/6),
            ('vary_dim/model_dim_-128_patt-rand_seed', 128, 3/6),
            ('vary_dim/model_dim_-256_patt-rand_seed', 256, 2/6),
            ('vary_len_b8/model_beak-8_patt-rand_seed',512,1/6),
         ],
        {'legendtitle': 'Number of\nneurons','tstar_type': 'log',

         # 'no_regr_models': {
         #     ('face', 'same'): [7,8],
         #     ('face', 'noisy'): [6,7, 8],
         #     ('rand', 'same'): [7, 8],
         # }
         },
    )


def test_prob(name_filter=''):
    plot_module(
        'test_prob',
        naming_unfolding('test_prob/model_prob-?_seed',
                         [('prob-?', ['0.2','0.4','0.6','0.8','1'], [0.2,0.4,0.6,0.8,1])]),
        {'legendtitle': 'Encoding Pr'}
    )


def big_simple_prob(name_filter=''):
    plot_module(
        'big_simple_prob',
        naming_unfolding('big_simple_prob/model_patt-rand_prob-?_seed',
                         [('prob-?', ['0.18','0.16','0.14','0.12','0.1', '0.005'], [0, 0.2,0.4,0.6,0.8,1])]),
        {'legendtitle': 'Encoding Pr'}
    )



def co_vary(name_filter=''):
    plot_module(
        'co_vary',
        [
         # ('co_vary/model_dim_-4_beak-1_patt-rand_seed', 4,  8 / 9),
         # ('co_vary/model_dim_-8_beak-2_patt-rand_seed', 8,  7 / 9),
            ('co_vary/model_dim_-16_beak-3_patt-rand_seed', 16,  7 / 7),
            ('co_vary/model_dim_-32_beak-4_patt-rand_seed', 32,  6 / 7),
            ('co_vary/model_dim_-64_beak-5_patt-rand_seed', 64,  5 / 7),
            ('co_vary/model_dim_-128_beak-6_patt-rand_seed', 128,  4 / 7),
            ('vary_len_b7/model_beak-7_patt-rand_seed', 256, 3 / 7),
            ('vary_len_b8/model_beak-8_patt-rand_seed', 512, 2 / 7),
            ('co_vary/model_dim_-1024_beak-9_patt-rand_seed', 1024, 1 / 7),
            ('co_vary/model_dim_-2048_beak-10_patt-rand_seed', 2048, 0 / 7),

            ('co_vary/model_dim_-16_beak-3_patt-face_seed', 16, 7 / 7),
            ('co_vary/model_dim_-32_beak-4_patt-face_seed', 32, 6 / 7),
            ('co_vary/model_dim_-64_beak-5_patt-face_seed', 64, 5 / 7),
            ('co_vary/model_dim_-128_beak-6_patt-face_seed', 128, 4 / 7),
            ('vary_len_b7/model_beak-7_patt-face_seed', 256, 3 / 7),
            ('vary_len_b8/model_beak-8_patt-face_seed', 512, 2 / 7),
            ('co_vary/model_dim_-1024_beak-9_patt-face_seed', 1024, 1 / 7),
            ('co_vary/model_dim_-2048_beak-10_patt-face_seed', 2048, 0 / 7),
         ],
        {'legendtitle': 'Number of\nneurons', 'tstar_type': 'log', 'init_type': 'log',
         'additional': ['ioSignal',],#'SNRratio'],
         'no_regr_models': {('face', 'noisy'): [2048]}
         #'no_curve_plot': True,
         }
    )



def co_vary_lowthre(name_filter=''):
    plot_module(
        'co_vary_lowthre',
        [
            ('co_vary/model_dim_-4_beak-1_patt-rand_seed', 4,  8 / 9),
            ('co_vary/model_dim_-8_beak-2_patt-rand_seed', 8,  7 / 9),
            ('co_vary/model_dim_-16_beak-3_patt-rand_seed', 16,  7 / 7),
            ('co_vary/model_dim_-32_beak-4_patt-rand_seed', 32,  6 / 7),
            ('co_vary/model_dim_-64_beak-5_patt-rand_seed', 64,  5 / 7),
            ('co_vary/model_dim_-128_beak-6_patt-rand_seed', 128,  4 / 7),
            ('vary_len_b7/model_beak-7_patt-rand_seed', 256, 3 / 7),

            ('co_vary/model_dim_-4_beak-1_patt-face_seed', 4, 8 / 9),
            ('co_vary/model_dim_-8_beak-2_patt-face_seed', 8, 7 / 9),
            ('co_vary/model_dim_-16_beak-3_patt-face_seed', 16, 7 / 7),
            ('co_vary/model_dim_-32_beak-4_patt-face_seed', 32, 6 / 7),
            ('co_vary/model_dim_-64_beak-5_patt-face_seed', 64, 5 / 7),
            ('co_vary/model_dim_-128_beak-6_patt-face_seed', 128, 4 / 7),
            ('vary_len_b7/model_beak-7_patt-face_seed', 256, 3 / 7),
         ],
        {'legendtitle': 'Number of\nneurons', 'tstar_type': 'log', 'init_type': 'log',

         #'no_curve_plot': True,
         }
    )

def test_sample_effect(name_filter=''):
    compared_models = [['test_sample_effect/model_samp-500_patt-face_seed', 500, 0],
                       ['test_sample_effect/model_samp-1000_patt-face_seed', 1000, 0.5],
                       ['test_sample_effect/model_samp-2000_patt-face_seed', 2000, 1],]
    plot_params = {'legendtitle': 'Sample'}
    plot_module('test_sample_effect', compared_models, plot_params)


def test_coding_f(name_filter=''):
    compared_models = [
                        #['test_coding_f/model_spar-True_inv_-2_seed', 2, 0.01],
                       ['test_coding_f/model_spar-False_seed', 0, 0.],
                       ['test_coding_f/model_spar-True_inv_-2_seed', 2, 0.1],
                       ['test_coding_f/model_spar-True_inv_-2.5_seed', 2.5, 0.4],
                       ['test_coding_f/model_spar-True_inv_-3_seed', 3, 0.5],
                       # ['test_coding_f/model_spar-True_inv_-4_seed', 4, 0.7],
                       # ['test_coding_f/model_spar-True_inv_-6_seed', 6, 0.8],
                       ['test_coding_f/model_spar-True_inv_-8_seed', 8, 0.9],
                       ]
    plot_params = {'legendtitle': 'Sparse'}
    plot_module('test_coding_f', compared_models, plot_params)


def test_neuron_num(name_filter=''):
    compared_models = [['test_neuron_num/model_dim_-16_samp-500_patt-face_seed', 16, 0],
                        ['test_neuron_num/model_dim_-16_samp-500_patt-face_seed', 16, 0],
                       ['test_neuron_num/model_dim_-32_samp-500_patt-face_seed', 32, 0.2],
                       ['test_neuron_num/model_dim_-64_samp-500_patt-face_seed', 64, 0.4],
                       ['test_neuron_num/model_dim_-128_samp-500_patt-face_seed', 128, 0.6],
                       ['test_neuron_num/model_dim_-256_samp-500_patt-face_seed', 256, 1],
    ]
    plot_params = {'legendtitle': 'Num'}
    plot_module('test_neuron_num', compared_models, plot_params)


def fair_comp(name_filter=''):
    compared_models = [['big_simple/model_patt-face_seed', 'Simple, q=1', 1],
                       ['big_simple/model_patt-rand_seed', 'Simple, q=1', 1],
                       ['big_simple_prob/model_patt-face_prob-0.128_seed', 'Simple, q=0.128', 0.7],
                       ['big_simple_prob/model_patt-rand_prob-0.128_seed', 'Simple, q=0.128', 0.7],
                       ['big_simple_prob/model_patt-face_prob-0.01_seed', 'Simple, q=0.01', 0.4],
                       ['big_simple_prob/model_patt-rand_prob-0.01_seed', 'Simple, q=0.01', 0.4],
                       ['vary_len_b8/model_beak-8_patt-face_seed', 'Complex, q=1', 0.0],
                       ['vary_len_b8/model_beak-8_patt-rand_seed', 'Complex, q=1', 0.0],]
    plot_params = {'legendtitle': '', 'tstar_type': 'bar'}
    plot_module('fair_comp', compared_models, plot_params)


def big_as_mel(name_filter=''):
    compared_models = [['big_as_mel/model_dim_-2862_beak-5_prob-1_seed', '5_prob-1', 0.0],
                       ['big_as_mel/model_dim_-2862_beak-5_prob-0.5_seed', '5_prob-0.5', 0.05],
                       # ['big_as_mel/model_dim_-2862_beak-5_prob-0.2_seed', '5_prob-0.2', 0.1],
                       # ['big_as_mel/model_dim_-2862_beak-5_prob-0.05_seed', '5_prob-0.05', 0.15],
                       # ['big_as_mel/model_dim_-2862_beak-5_prob-0.02_seed', '5_prob-0.02', 0.2],
                       ['big_as_mel/model_dim_-2612_beak-6_prob-1_seed', '6_prob-1', 0.4],
                       ['big_as_mel/model_dim_-2612_beak-6_prob-0.5_seed', '6_prob-0.5', 0.45],
                       ]
    plot_params = {'legendtitle': ''}
    plot_module('big_as_mel', compared_models, plot_params)



def big_as_mel_match(name_filter=''):
    compared_models = [
                       #  ['big_as_mel_match/model_dim_-1012_beak-5_prob-1_seed', '5_prob-1', 0.0],
                       # ['big_as_mel_match/model_dim_-1012_beak-5_prob-0.5_seed', '5_prob-0.5', 0.1],
                       # ['big_as_mel_match/model_dim_-923_beak-6_prob-1_seed', '6_prob-1', 0.4],
                       # ['big_as_mel_match/model_dim_-923_beak-6_prob-0.5_seed', '6_prob-0.5', 0.5],
                        ['big_as_mel_match/model_dim_-2262_beak-1_prob-1_seed', '1_prob-1', 0.0],
                        ['big_as_mel_match/model_dim_-2262_beak-1_prob-0.5_seed', '1_prob-0.5', 0.2],
                        ['big_as_mel_match/model_dim_-2262_beak-1_prob-0.2_seed', '1_prob-0.2', 0.4],
                        ['big_as_mel_match/model_dim_-2262_beak-1_prob-0.1_seed', '1_prob-0.1', 0.6],

                        # ['big_as_mel_match/model_dim_-1600_beak-2_prob-1_seed', '2_prob-1', 0.2],
                        # ['big_as_mel_match/model_dim_-1600_beak-2_prob-0.5_seed', '2_prob-0.5', 0.2],
                        # ['big_as_mel_match/model_dim_-1600_beak-2_prob-0.2_seed', '2_prob-0.2', 0.2],
                        # ['big_as_mel_match/model_dim_-1600_beak-2_prob-0.1_seed', '2_prob-0.1', 0.2],
                        #
                        # ['big_as_mel_match/model_dim_-1306_beak-3_prob-1_seed', '3_prob-1', 0.4],
                        # ['big_as_mel_match/model_dim_-1306_beak-3_prob-0.5_seed', '3_prob-0.5', 0.4],
                        # ['big_as_mel_match/model_dim_-1306_beak-3_prob-0.2_seed', '3_prob-0.2', 0.4],
                        # ['big_as_mel_match/model_dim_-1306_beak-3_prob-0.1_seed', '3_prob-0.1', 0.4],
                        #
                        # ['big_as_mel_match/model_dim_-1131_beak-4_prob-1_seed', '4_prob-1', 0.6],
                        # ['big_as_mel_match/model_dim_-1131_beak-4_prob-0.5_seed', '4_prob-0.5', 0.6],
                        # ['big_as_mel_match/model_dim_-1131_beak-4_prob-0.2_seed', '4_prob-0.2', 0.6],
                        # ['big_as_mel_match/model_dim_-1131_beak-4_prob-0.1_seed', '4_prob-0.1', 0.6],
                       ]
    plot_params = {'legendtitle': ''}
    plot_module('big_as_mel_match', compared_models, plot_params)