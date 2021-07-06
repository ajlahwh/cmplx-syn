import numpy as np
import roblib
import matplotlib.pyplot as plt

for beak_num in [6,5,4,3,2,1]:
    d = roblib.load(f'files/simulations/monitor_snr/model_beak-{beak_num}_patt-rand_seed-0/monitor_simulation.bk')

    # d = roblib.load('files/simulations/co_vary/model_dim_-32_beak-4_patt-rand_seed-0/simulation.bk')
    # plt.plot(d['times_famil'][:70],d['r_signal_famil'][:70,0,:].mean(-1))
    # plt.xlabel('time')
    # plt.ylabel('readout signal')
    # plt.title('dim_-32_beak-4')
    # plt.show()

    min_present_time = 100
    for idx in range(len(d['present_time'])):
        pt = np.diff(d['present_time'][idx])
        min_present_time = min(min_present_time, len(pt))
        if idx == -1:
            plt.figure()
            plt.subplot(3, 1, 1)
            plt.title(f'beak_num{beak_num}')
            plt.plot(d['io_signal_famil'][idx], alpha=0.5)
            plt.vlines(d['present_time'][idx], 0, 1, 'r')
            plt.ylabel('io_signal')
            plt.subplot(3, 1, 2)
            plt.plot(d['r_signal_famil'][idx], alpha=0.5)
            plt.vlines(d['present_time'][idx], 0, 1, 'r')
            plt.ylabel('r_signal')
            plt.subplot(3, 1, 3)
            plt.plot(pt)
            plt.ylabel('interval')
            plt.show()

    avg_present_time = np.array([np.diff(d['present_time'][idx])[:min_present_time] for idx in range(10)])
    avg_present_time = avg_present_time.mean(0)
    plt.plot(avg_present_time, label=beak_num)
plt.ylabel('interval between presentation')
plt.xlabel('re-present times')
plt.legend()
plt.show()