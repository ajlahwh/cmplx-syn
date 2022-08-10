import numpy as np
import roblib
import matplotlib.pyplot as plt
import joblib
import time
import matplotlib.patches as mpatches
from plot_maker import set_mpl, plot_start
import matplotlib
cmap = matplotlib.cm.get_cmap('plasma')
init_signal= 1.3159186857273735
thre = 0.5

# plt.figure()
# for b in range(1,9):
#     a = roblib.load(f'files/simulations/vary_len_b8/model_beak-{b}_patt-rand_seed')
#     iosignal, time = a.iloc[0]['perf'], a.iloc[0]['time']
#     plt.loglog(time + 1, iosignal, label=b)
# plt.loglog(time+1, init_signal/np.sqrt(time+1),label='ideal')
# plt.legend()
# plt.show()
#
# plt.figure()
# for n in [256]: #16,32,64,128,
#     a = roblib.load(f'files/simulations/vary_dim/model_dim_-{n}_patt-rand_seed')
#     iosignal, time = a.iloc[0]['perf'], a.iloc[0]['time']
#     plt.loglog(time + 1, iosignal, label=n)
# plt.loglog(time+1, init_signal/np.sqrt(time+1),label='ideal')
# plt.xlim([0.5,10**6])
# plt.legend()
# plt.show()

class Hyper(object):
    name = 'Hyperbolic'
    def __init__(self, t_init):
        self.t_init = t_init
        self.init_signal = init_signal
    def __call__(self, t):
        s = self.init_signal/ (t-self.t_init + 1)
        return s

class Plaw(object):
    name = 'Inverse-square-root'
    def __init__(self, t_init):
        self.t_init = t_init
        self.init_signal = init_signal

    def __call__(self, t):
        s = self.init_signal/ np.sqrt((t-self.t_init) + 1)
        return s

class Exp(object):
    name = 'Exponential'
    def __init__(self, t_init, tau=7.486):
        self.t_init = t_init
        self.init_signal = 1
        self.tau = tau

    def __call__(self, t):
        s = self.init_signal/ np.exp((t-self.t_init)/self.tau)
        return s

def get_optimal_present_time(max_count=100, ideal_model=None):
    assert ideal_model is not None
    present_time_count = 0
    presentation_times = []
    total_s = 0
    total_s_history = []
    additive_traces = []
    cur_t = 0
    start = time.time()
    while present_time_count<max_count:
        if total_s < thre:
            present_time_count += 1
            additive_traces.append(ideal_model(cur_t))
            presentation_times.append(cur_t)
            if present_time_count % 25 ==0:
                print(present_time_count,time.time()-start, additive_traces[0](cur_t))
        total_s = np.sum([h(cur_t) for h in additive_traces])
        total_s_history.append(total_s)
        cur_t += 1
    total_t_history = np.arange(cur_t) + 1
    return presentation_times, total_s_history



def fixed_present_ideal_model(gamma_values, beta_values, max_present_time=100, ideal_model=None):
    gamma_point_num = len(gamma_values)
    beta_point_num =  len(beta_values)
    assert ideal_model is not None
    signal_mat = np.zeros([max_present_time, gamma_point_num, beta_point_num])
    for gamma_idx in range(gamma_point_num):
        for beta_idx in range(beta_point_num):
            gamma = gamma_values[gamma_idx]
            beta = beta_values[beta_idx]
            interval_to_present = lambda k: gamma * k**beta
            cur_t = 0
            additive_traces = []
            additive_traces.append(ideal_model(cur_t))
            for pt in range(1, max_present_time):
                cur_t += int(np.round(interval_to_present(pt)))
                total_s = np.sum([h(cur_t) for h in additive_traces])
                signal_mat[pt, gamma_idx, beta_idx] = total_s
                additive_traces.append(ideal_model(cur_t))

    return signal_mat


def diagram_fixed_ideal_model():
    set_mpl()
    for schedule, interval_to_present in [
        ('constant', lambda k: M/4),
        ('linear',lambda k: M/4 * k),
    ]:
        for ideal_model in [Plaw, Exp, Hyper]:
            cur_t = 0
            next_t = 0
            pt = 0
            additive_traces = []
            ideal_present_time = [0]
            total_s_history = []
            while pt<=10:
                if cur_t == next_t:
                    additive_traces.append(ideal_model(cur_t))
                    pt += 1
                    interval = int(np.ceil(interval_to_present(pt)))
                    next_t = cur_t + interval
                    ideal_present_time.append(next_t)
                total_s = np.sum([h(cur_t) for h in additive_traces])
                total_s_history.append(total_s)
                cur_t += 1
            pretty_plot_traces(total_s_history, ideal_present_time,
                           save_fname=f'{schedule}_schedule_{ideal_model.__name__}', title=f'{ideal_model.name} decay')
            pretty_plot_intervals(ideal_present_time,
                           save_fname=f'{schedule}_schedule_{ideal_model.__name__}')


def diagram_optimal_ideal_model():
    set_mpl()
    for ideal_model in [Plaw, Exp,
                        Hyper
                        ]:
        ideal_present_time, total_s_history = get_optimal_present_time(10, ideal_model=ideal_model)
        pretty_plot_traces(total_s_history, ideal_present_time,
                           save_fname=f'optimal_schedule_{ideal_model.__name__}', title=f'{ideal_model.name} decay', threshold=0.5)
        pretty_plot_intervals(ideal_present_time,
                           save_fname=f'optimal_schedule_{ideal_model.__name__}')



def pretty_plot_traces(total_s_history, ideal_present_time, save_fname='', title='', threshold=None):
    plot_start(square=True)
    plt.plot(total_s_history, linewidth=0.5)
    ax = plt.gca()
    for x in ideal_present_time:
        arrow = mpatches.FancyArrowPatch((x, 4), (x, 3),
                                         mutation_scale=4, color='r', arrowstyle='wedge')
        ax.add_patch(arrow)
        if x>0 and x-1<len(total_s_history):
            plt.scatter(x-1, total_s_history[x-1], color='orange', marker='*', s=5)
        # plt.arrow(x, 2, 10, 0, width=0.1, head_width=0.5)
    if threshold is not None:
        plt.hlines(threshold, 0, ideal_present_time[-1], 'k', '--', linewidth=0.5)
    plt.xlim([-2, ideal_present_time[-1] + 2])
    # plt.ylim([-0.3, 0.1+max(2, np.max(total_s_history))])
    # plt.yticks([0, 1, 2])
    plt.ylim([10**-1, 5])
    plt.yticks([10**-1, 1])
    plt.title(title)
    plt.xlabel('Time since first presentation')
    plt.ylabel('ioSignal')
    ax.set_yscale('log')
    plt.savefig(f'figures/{save_fname}_signal.pdf', bbox_inches="tight")
    plt.show()
    plt.close()

def pretty_plot_intervals(ideal_present_time, save_fname='', title=''):
    plot_start(square=True)
    intervals = np.diff(ideal_present_time)
    plt.plot(np.arange(len(intervals)) + 1, intervals, linewidth=0.5)
    from scipy.stats import linregress
    slope, intercept, r, p, se = linregress(np.log(np.arange(len(intervals)) + 1), intervals)
    print(slope, intercept, r, p, se)
    # plt.xlim([0, len(intervals)])
    # if intervals.max()<=20:
    #     plt.ylim([0,20])
    # else:
    #     plt.ylim([0,200])
    # plt.xticks([0, 5, 10])
    plt.xlabel('Interval number')
    plt.ylabel('Interval')
    ax = plt.gca()
    ax.set_xscale('log')
    # ax.set_yscale('log')
    plt.savefig(f'figures/{save_fname}_interval.pdf', bbox_inches="tight")
    plt.show()
    plt.close()


def get_ideal_model_interval(ideal_model=None):
    ideal_present_time, _ = get_optimal_present_time(1000, ideal_model=ideal_model)
    interval = np.diff(ideal_present_time)
    joblib.dump(interval, f'files/ideal_present_time_{ideal_model.__name__}.pkl')

A = np.pi/np.sqrt(2)
M = (A*init_signal/thre)**2

max_present_time = 260

def get_signal_rpt_mat(gamma_values, beta_values):
    gamma_point_num = len(gamma_values)
    beta_point_num =  len(beta_values)
    plaw_signal_mat = fixed_present_ideal_model(gamma_values, beta_values, max_present_time=max_present_time, ideal_model=Plaw)
    exp_signal_mat = fixed_present_ideal_model(gamma_values, beta_values, max_present_time=max_present_time, ideal_model=Exp)
    hyper_signal_mat = fixed_present_ideal_model(gamma_values, beta_values, max_present_time=max_present_time, ideal_model=Hyper)
    joblib.dump([plaw_signal_mat, exp_signal_mat, hyper_signal_mat],
                f'signal_rpt_mat_gamma{gamma_point_num}_beta{beta_point_num}.pkl')


def pretty_signal_gain_curve():
    gamma_values = np.concatenate(
        [np.arange(1, 10, 1), np.arange(10, 100, 10), np.arange(100, 1000, 100), np.arange(1000, 10000, 1000), ], 0)
    gamma_point_num = len(gamma_values)
    beta_point_num = 21
    beta1_loc = beta_point_num // 2
    beta_values = np.linspace(0, 2, num=beta_point_num, endpoint=True)
    # get_signal_rpt_mat(gamma_values, beta_values)
    plaw_signal_mat, exp_signal_mat, hyper_signal_mat = joblib.load(
                f'signal_rpt_mat_gamma{gamma_point_num}_beta{beta_point_num}.pkl')


    plot_start(square=False, figsize=(3, 1.5))
    plt.loglog(gamma_values, exp_signal_mat[1,:,0],label='Exponential')
    plt.loglog(gamma_values, plaw_signal_mat[1,:,0],label='Inverse-square-root')
    plt.loglog(gamma_values, hyper_signal_mat[1,:,0],label='Hyperbolic')
    plt.xlabel('Coefficient gamma')
    plt.ylabel('io Signal (moment when #Int=1)')
    plt.ylim([10**-5, 2])
    plt.legend()
    plt.savefig(f'figures/signal_init_pres.pdf', bbox_inches="tight")
    plt.show()

    for ideal_model in ['plaw', 'exp', 'hyper']:
        for schedule in ['const', 'linear']:
            beta_loc, ylabel = {'const': (0, 'Constant schedule'), 'linear': (beta1_loc, 'Linear schedule')}[schedule]
            signal_mat, title = {
                'plaw': (plaw_signal_mat, 'Inverse-square-root'),
                'exp': (exp_signal_mat, 'Exponential'),
                'hyper': (hyper_signal_mat, 'Hyperbolic')}[ideal_model]

            plot_start(square=True)
            for t in range(2, max_present_time):
                if t in [2,4,8,16,32,64, 128, 256]:
                    label=t
                else:
                    label=''
                plt.semilogx(gamma_values, np.log(signal_mat[t,:,beta_loc])-np.log(signal_mat[1,:,beta_loc]),
                           color=cmap(np.log(t)/np.log(max_present_time)),label=label)
            plt.xlabel('Coefficient gamma')
            plt.ylabel(ylabel+'\nSignal gain rel. to #Int=1 (Bel)')
            # leg = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=False, ncol=1)
            # leg.set_title('#Int')
            plt.ylim([-5, 5])
            plt.yticks([-5,0,5])
            plt.hlines(0, gamma_values[0], gamma_values[-1],'k','--')
            plt.xlim([gamma_values[0], gamma_values[-1]])
            plt.xticks([10**0,10**2,10**4])
            plt.title(title)

            plt.savefig(f'figures/{schedule}_{ideal_model}_gain.pdf', bbox_inches="tight")
            plt.show()
            plt.close()


def pretty_signal_gain_square():
    gamma_point_num = 51
    gamma_values = np.logspace(0, 4, num=gamma_point_num, endpoint=True, base=10)
    gamma_ticks = [1,10**2,10**4]

    beta_point_num = 51
    beta1_loc = beta_point_num // 2
    beta_values = np.linspace(0, 2, num=beta_point_num, endpoint=True)
    beta_ticks = [0, 1, 2]

    # get_signal_rpt_mat(gamma_values, beta_values)
    plaw_signal_mat, exp_signal_mat, hyper_signal_mat = joblib.load(
        f'signal_rpt_mat_gamma{gamma_point_num}_beta{beta_point_num}.pkl')

    def wrap_plot_all_t(sig_mat, ylabel):
        for t_idx, t in enumerate([2,8,32,128]):#range(0, max_present_time, plot_interval):
            plt.subplot(1, 4, t_idx + 1)
            cur_signal = np.log10(sig_mat[t] + 1e-500) - np.log10(sig_mat[1] + 1e-500)
            thre = 0.2
            cur_signal[cur_signal > thre] = 1
            cur_signal[cur_signal < -thre] = -1
            cur_signal[np.logical_and(cur_signal != 1, cur_signal != -1)] = 0
            # cur_signal[np.logical_and(cur_signal!=-1, cur_signal!=1)] = 0
            plt.imshow(cur_signal, vmin=-1, vmax=1, cmap='coolwarm')
            plt.xlabel(f'#Int={t}\nExponent beta', rotation=0)

            if t_idx == 0:
                plt.yticks([0, gamma_point_num//2, gamma_point_num-1], ['$10^0$','$10^2$','$10^4$'], rotation=0)
                plt.ylabel(ylabel+'\nCoefficient gamma')
            else:
                plt.yticks([0, gamma_point_num//2, gamma_point_num-1],['','',''])
            # if t_idx ==7:
                # plt.xlabel('')
            plt.xticks([0, beta_point_num//2, beta_point_num-1], beta_ticks, rotation=0)
            # else:
            # plt.xticks([0, beta_point_num//2, beta_point_num-1],['','',''])

    plot_start(square=False, figsize=(4, 1.5))
    wrap_plot_all_t(plaw_signal_mat, 'Inverse-square-root')
    plt.savefig(f'figures/fixed_square_plaw.pdf', bbox_inches="tight")
    plt.show()
    plt.close()

    plot_start(square=False, figsize=(4, 1.5))
    wrap_plot_all_t(exp_signal_mat, 'Exponential')
    plt.savefig(f'figures/fixed_square_exp.pdf', bbox_inches="tight")
    plt.show()
    plt.close()

    plot_start(square=False, figsize=(4, 1.5))
    wrap_plot_all_t(hyper_signal_mat, 'Hyperbolic')
    plt.savefig(f'figures/fixed_square_hyper.pdf', bbox_inches="tight")
    plt.show()
    plt.close()

if __name__ == '__main__':
    diagram_optimal_ideal_model()
    # diagram_fixed_ideal_model()
    # get_ideal_model_interval(ideal_model=Exp)
    # get_ideal_model_interval(ideal_model=Plaw)
    # get_ideal_model_interval(ideal_model=Hyper)
    # pretty_signal_gain_curve()
    pretty_signal_gain_square()

    if 0:##### limiting value of M
        ideal_present_time = joblib.load('files/ideal_present_time.pkl')
        # print(ideal_present_time/(np.arange(ideal_present_time.size)+1))
        plt.plot((ideal_present_time/(np.arange(ideal_present_time.size)+1)), label='simulated: [ideal interval tau_k]/k')
        plt.hlines(M, 1, ideal_present_time.size,label='theoretic: M=(A_infty*C/theta)^2', color='k')
        plt.legend(loc='lower center')
        plt.xlabel('presentation times')
        plt.ylabel('linear scaling coefficient')
        plt.show()

        # plt.figure()
        # plt.subplot(1,2,1)
        # plt.plot(np.arange(ideal_present_time.size)+1, ideal_present_time)
        # plt.ylabel('ideal interval')
        # plt.xlabel('presentation times')
        # plt.subplot(1,2,2)
        # plt.loglog(np.arange(ideal_present_time.size)+1, ideal_present_time)
        # plt.xlabel('presentation times')
        # plt.show()
        # plt.figure()

    if 0:##### limiting value of A_k
        coef = 1 + 0.1
        fig, ax = plt.subplots()
        for k in [10,100,1000,10000,100000,1000000]: #[100000000,10000000,1000000,100000,10000,1000,100,10]:
            # A_k = np.sum([1 / np.sqrt((2*k-i)*(i+1)/2) for i in range(k)])
            A_k = np.sum([1 / np.sqrt(np.sum([(k-j)**coef for j in range(i+1)])) for i in range(k)])
            T = (A_k*init_signal / 0.5)**2
            print(k, A_k, T)
            plt.scatter(k,A_k)
        ax.set_xscale('log')
        # plt.hlines(A, 10, 100000000,label='theoretic limit A_infty')
        plt.xlabel('k terms in summation')
        plt.ylabel('A_k')
        plt.legend()
        plt.show()

    if 0:
        # for b in range(1, 8):
        #     a = roblib.load(f'files/simulations/vary_len_b8/model_beak-{b}_patt-rand_seed')
        #     iosignal, time = a.iloc[0]['perf'], a.iloc[0]['time']
        #     plt.loglog(time + 1, iosignal, label=f'N=512, m={b}', alpha=0.3)

        b = 8
        a = roblib.load(f'files/simulations/vary_len_b8/model_beak-{b}_patt-rand_seed')
        iosignal, time = a.iloc[0]['perf'], a.iloc[0]['time']
        plt.loglog(time + 1, iosignal, label=f'N=512, m={b}')

        # a = roblib.load('files/simulations/co_vary/model_dim_-1024_beak-9_patt-face_seed')
        # iosignal, time = a.iloc[0]['perf'], a.iloc[0]['time']
        # plt.loglog(time + 1, iosignal, label=f'N=1024, m=9')
        #
        # a = roblib.load('files/simulations/co_vary/model_dim_-2048_beak-10_patt-face_seed')
        # iosignal, time = a.iloc[0]['perf'], a.iloc[0]['time']
        # plt.loglog(time + 1, iosignal, label=f'N=2048, m=10')

        a = roblib.load('files/simulations/fixed_len_b8_level128/model_beak-8_patt-rand_seed')
        iosignal, time = a.iloc[0]['perf'], a.iloc[0]['time']
        plt.loglog(time + 1, iosignal, label=f'N=512, m=8, 128levels')

        plt.loglog(time + 1, 1.3159186857273735 / np.sqrt(time + 1), label=f'ideal square-root model')

        b = 1
        idx = 40
        a = roblib.load(f'files/simulations/vary_len_b8/model_beak-{b}_patt-rand_seed')
        iosignal, time = a.iloc[0]['perf'], a.iloc[0]['time']
        plt.semilogy(time[:idx], iosignal[:idx], label=f'N=512, m={b}')
        plt.semilogy(time[:idx], 1 / np.exp(time[:idx] / 7.486), label=f'ideal exponential model')
        plt.legend()
        plt.show()