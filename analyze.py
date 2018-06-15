import sys
import matplotlib.pyplot as plt

import diff_calc
from mods import str_to_mods


iter_colors = iter(["#0ED4F9","#83BAFA","#BF9CE1","#E6759F","#D66759"])


if __name__ == "__main__":
    
    name = sys.argv[1]

    if len(sys.argv) >= 3:
        mods_str = sys.argv[2]
    else:
        mods_str = '-'

    miss_probs, IPs, times, IPs_raw, IPs_corr0, strain_history = diff_calc.analyze_file_diff('data/maps/' + name + '.json', str_to_mods(mods_str))

    
    fig, axarr = plt.subplots(2, sharex=True)
    
    axarr[0].plot(times, IPs, '.', alpha=0.8)
    axarr[0].vlines(times, IPs_raw, IPs_corr0, colors=(1.0,0.5,0.5,0.8), linewidths=1)
    axarr[0].vlines(times, IPs_corr0, IPs, colors=(0.3,1.0,0.3,0.8), linewidths=1)

    # ax_corr = axarr[0].twinx()
    # ax_corr.set(ylim=[0, 1])
    # ax_corr.plot(times, corr0s, 'r.', times, corr3s, 'g.', markersize=2, alpha=0.3)
    # ax_corr.fill_between(times, corr0s, corr3s, where=corr3s>corr0s)

    axarr[1].plot(times, miss_probs, '.', alpha=0.8)

    plt.show()


    fig, ax = plt.subplots()


    
    strains = [x[0] for x in strain_history]
    times_tap = [x[1] for x in strain_history]
    transposed_strains = [list(x) for x in zip(*strains)]

    for s in transposed_strains:
        ax.plot(times_tap, s, '.-', markersize=3, color=next(iter_colors))

    # strains0 = [x[0][0] for x in strain_history]
    # strains1 = [x[0][1] for x in strain_history]
    # strains2 = [x[0][2] for x in strain_history]
    # times_tap = [x[1] for x in strain_history]
    # ax.plot(times_tap, strains0, '.-',
    #         times_tap, strains1, '.-',
    #         times_tap, strains2, '.-',
    #         markersize=3)

    # print(strain_history)

    plt.show()
