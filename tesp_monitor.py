import numpy as np
import roblib
import matplotlib.pyplot as plt
from plot_maker import get_reg_coef
import joblib
from naive_monitor import pretty_plot_traces
import matplotlib
cmap = matplotlib.cm.get_cmap('plasma')
from plot_maker import plot_start
save_path = 'monitor_snr_500_128levels'
var = [8, 7, 6, 5, 4, 3, 2, 1][::-1]


def present_time_combine_all_seeds():
    for beak_num in var:
        d = {}
        for seed in range(10):
            dd = roblib.load(
                f'files/simulations/{save_path}/model_beak-{beak_num}_patt-rand_seed-{seed}/monitor_simulation.bk')
            for k in dd.keys():
                if k not in d:
                    d[k] = []
                d[k] += dd[k]
        joblib.dump((d['present_time'], d['io_signal_famil']),
                    f'files/simulations/{save_path}/model_beak-{beak_num}_patt-rand_monitor_presenttime.bk')


def pretty_traces():
    for beak_num in [8, 1]:
        present_time_traces, io_signal_famil = joblib.load(
            f'files/simulations/{save_path}/model_beak-{beak_num}_patt-rand_monitor_presenttime.bk')
        present_numbers = 10
        for idx in range(1):
            present_time = present_time_traces[idx]
            time_last_number = present_time[present_numbers + 1]
            total_s_history = io_signal_famil[idx][:time_last_number + 1]

            pretty_plot_traces(
                total_s_history,
                present_time[:present_numbers + 1],
                save_fname=f'optimal_schedule_beak-{beak_num}', title=f'N={64}, m={beak_num}')


def pretty_all_ideal_interval():
    ideal_present_time_plaw = joblib.load('files/ideal_present_time_plaw.pkl')
    ideal_present_time_exp = joblib.load('files/ideal_present_time_exp.pkl')
    ideal_present_time_hyper = joblib.load('files/ideal_present_time_hyper.pkl')
    fig, ax = plot_start(square=True)
    leng = len(ideal_present_time_plaw)
    ax.plot(1 + np.arange(leng), ideal_present_time_exp, label='Exponential')
    ax.plot(1 + np.arange(leng), ideal_present_time_plaw, label='Inverse-square-root')
    ax.plot(1 + np.arange(leng), ideal_present_time_hyper, label='Hyperbolic')
    ax.set_ylabel('Interval')
    ax.set_xlabel('Interval number')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.xlim([1, 1000])
    plt.ylim([1, 5*10**4])
    plt.legend(loc='upper center')
    plt.savefig(f'figures/interval_ideal_rand-same.pdf', bbox_inches="tight")
    plt.show()
    plt.close()


def pretty_all_beakers_interval():
    ideal_present_time_plaw = joblib.load('files/ideal_present_time_plaw.pkl')
    ideal_present_time_exp = joblib.load('files/ideal_present_time_exp.pkl')

    limit_list = []
    turn_point_list = []
    fig, ax = plot_start(square=False, figsize=(3,3))
    for beak_num in var:
        present_time_traces, _ = joblib.load(
            f'files/simulations/{save_path}/model_beak-{beak_num}_patt-rand_monitor_presenttime.bk')
        pt_list = []
        num_patterns = len(present_time_traces)
        for idx in range(num_patterns):
            cur_pt = np.diff(present_time_traces[idx])
            pt_list.append(cur_pt)
        pt_patterns = np.array(pt_list)
        avg_present_time = np.mean(pt_patterns, 0)  # np.nanmean(pt_patterns, 0)
        color = cmap((beak_num-1)/7.0)
        plt.plot(1 + np.arange(avg_present_time.size), avg_present_time, label=beak_num, color=color)
        limit = avg_present_time[-30:].mean()
        # plt.hlines(limit, 1, avg_present_time.size, color=color, linestyle='--', alpha=0.5)
        turn_point = np.argwhere(avg_present_time > limit * 0.99)[0, 0]
        # plt.vlines(1+turn_point, 0, limit + 1, color=color, linestyle='--', alpha=0.5)
        limit_list.append(limit)
        turn_point_list.append(turn_point)

    max_t = 500
    ax.plot(1 + np.arange(max_t), ideal_present_time_plaw[:max_t], color='k',linestyle='dotted',alpha=1)
    ax.plot(1 + np.arange(max_t), ideal_present_time_exp[:max_t], color='k',linestyle='dotted',alpha=1)
    plt.fill_between(1 + np.arange(max_t), ideal_present_time_exp[:max_t], ideal_present_time_plaw[:max_t], color='k', alpha=0.1)
    ax.text(0.2, 0.15, 'Exponential decay', transform=ax.transAxes)
    ax.text(0.2, 0.55, 'Inverse-square-root decay', transform=ax.transAxes, rotation=40)
    ax.set_ylabel('Interval')
    ax.set_xlabel('Interval number')
    leg = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=False, ncol=1)
    leg.set_title('Synaptic\ncomplexity')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.xlim([1, max_t])
    plt.ylim([1, 10**4])
    plt.savefig(f'figures/interval_rand-same.pdf', bbox_inches="tight")
    plt.show()
    plt.close()
    joblib.dump([limit_list, turn_point_list],'all_beaker_scaling.pkl')
    

def pretty_scaling_beakers():
    limit_list, turn_point_list = joblib.load('all_beaker_scaling.pkl')
    for plot_params in [
        (limit_list, [1, 10 ** 4], [1, 10 ** 2, 10 ** 4],'Interval when saturated', 'interval'),
        (turn_point_list, [1, 10 ** 3], [1, 10,100, 10 ** 3], 'Interval number when saturated', 'ptnum'),
    ]:
        values, ylim, yticks, ylabel, fname = plot_params
        fig, ax = plot_start(square=True)
        coef, regr, score = get_reg_coef(var,
                                         np.log2(values),
                                         get_reg=True)
        y_pred = 2 ** (regr(var))
        plt.semilogy(var, y_pred, alpha=1, linestyle='--', color='C0', marker=None, label=f'{coef:.2f}')
        plt.semilogy(var, values, alpha=0.7, linestyle='None',
                     color='C0', marker='x', markersize=6)
        plt.xlim([0, 9])
        plt.xticks([2, 4, 6, 8])
        plt.xlabel('Synaptic complexity')
        plt.ylim(ylim)
        plt.yticks(yticks)
        plt.ylabel(ylabel)
        plt.legend(loc='upper left')#, bbox_to_anchor=(1, 0.5), fancybox=True, shadow=False, ncol=1)
        plt.savefig(f'figures/{fname}_saturated_linear.pdf', bbox_inches="tight")
        plt.show()
        plt.close()


if __name__ == '__main__':
    pass
    # present_time_combine_all_seeds()
    # pretty_traces()
    # pretty_all_ideal_interval()
    # pretty_all_beakers_interval()
    # pretty_scaling_beakers()