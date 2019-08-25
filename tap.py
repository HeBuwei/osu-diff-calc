import sys


import numpy as np
from scipy.special import expit

from aim import cs_to_diameter, calc_distance



def calc_tap_diff(beatmap, analysis=False):
    
    # k = np.array([0.3, 1.5, 7.5])
    # k = np.exp(np.linspace(1.6, -2, num=9))
    k = np.exp(np.linspace(1.7, -1.6, num=4))

    diameter = cs_to_diameter(beatmap["CsAfterMods"])
    hit_objects = beatmap['hitObjects']
    curr_strain = k * 1
    prev_time = hit_objects[0]['startTime'] / 1000
    max_strain = k * 1
    strain_history = [(list(curr_strain), prev_time)]

    for obj0, obj1 in zip(hit_objects, hit_objects[1:]):
        
        curr_time = obj1['startTime'] / 1000
        curr_strain *= np.exp(-k * (curr_time - prev_time))

        if analysis:
            strain_history.append((list(curr_strain), curr_time))

        max_strain = np.maximum(max_strain, curr_strain)
        obj1['tapStrain'] = curr_strain.copy()

        d = calc_distance(obj0['position'], obj1['position'])
        spaced_buff = calc_spacedness(d / diameter) * 0.07

        curr_strain += k * (1 + spaced_buff)
        prev_time = curr_time

    diff = np.average(max_strain) ** 0.85 * 0.758

    if analysis:
        return strain_history
    else:
        return (diff,) + tuple(max_strain)



def calc_tap_diff_for_pp(beatmap):

    k = np.exp(np.linspace(1.7, -1.6, num=4))
    mash_levels = np.linspace(0, 1, num=6)

    hit_objects = beatmap['hitObjects']
    prev_time = hit_objects[0]['startTime'] / 1000
    curr_strain = np.zeros((len(mash_levels), len(k))) + k
    max_strain = np.zeros((len(mash_levels), len(k))) + k

    diameter = cs_to_diameter(beatmap["CsAfterMods"])

    for obj0, obj1 in zip(hit_objects, hit_objects[1:]):
        
        curr_time = obj1['startTime'] / 1000
        curr_strain *= np.exp(-k * (curr_time - prev_time))
        max_strain = np.maximum(max_strain, curr_strain)

        obj1['tapStrain'] = curr_strain.copy()
        d = calc_distance(obj0['position'], obj1['position'])

        nerf_factors = calc_mash_nerf_factors(mash_levels, d / diameter)

        curr_strain +=  nerf_factors[:,np.newaxis] * k
        prev_time = curr_time


    return max_strain

    # diff = np.average(max_strain) ** 0.85 * 0.778



def calc_mash_nerf_factors(mash_levels, d_relative):
    
    complete_mash_factors = 0.5 + 0.5 * expit(d_relative * 7 - 6)
    return mash_levels * complete_mash_factors + (1 - mash_levels) * 1


def calc_spacedness(d_relative):
	return expit((d_relative - 0.4) * 10) - expit(-4)



if __name__ == '__main__':

    from mods import str_to_mods
    from beatmap import load_beatmap
    from diff_calc import remove_spinners, apply_mods

    
    filepath = sys.argv[1]

    if len(sys.argv) >= 3:
        mods_str = sys.argv[2]
    else:
        mods_str = '-'

    beatmap = load_beatmap(filepath)
    remove_spinners(beatmap)
    apply_mods(beatmap, str_to_mods(mods_str))
    
    tap_diff_for_pp = calc_tap_diff_for_pp(beatmap)

    print(np.mean(tap_diff_for_pp,axis=1))

    