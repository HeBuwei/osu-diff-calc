import sys

import numpy as np
from scipy.special import expit
from scipy.linalg import norm

from mods import str_to_mods
from beatmap import load_beatmap
from fitts_law import calc_IP, calc_hit_prob
from aim import calc_aim_diff, cs_to_diameter




def calc_file_diff(file_path, mods=["nm", "nm"], tap_detail=False):
    beatmap = load_beatmap(file_path)
    return calc_diff(beatmap, mods, tap_detail)


def calc_diff(beatmap, mods=["nm", "nm"], tap_detail=False):
    
    remove_spinners(beatmap)
    apply_mods(beatmap, mods)
    
    tap_diff = calc_tap_diff(beatmap)
    aim_diff = calc_aim_diff(beatmap)

    i = 7.0
    overall_diff = (aim_diff ** i + tap_diff[0] ** i) ** (1/i) * 0.968

    if tap_detail:
        return (aim_diff,) + tap_diff + (overall_diff,)
    else:
        return (aim_diff,) + tap_diff[:1] + (overall_diff,)


def analyze_file_diff(file_path, mods=["nm", "nm"]):
    beatmap = load_beatmap(file_path)
    return analyze_diff(beatmap, mods)


def analyze_diff(beatmap, mods=["nm", "nm"]):

    remove_spinners(beatmap)
    apply_mods(beatmap, mods)
    
    cs = float(beatmap['CircleSize'])
    diameter = cs_to_diameter(cs)

    tap_strains = calc_tap_diff(beatmap, analysis=True)
    TP, movements = calc_aim_diff(beatmap, analysis=True)

    miss_probs = [1 - calc_hit_prob(mvmt.D, diameter, mvmt.MT, TP) for mvmt in movements]
    IPs = [calc_IP(mvmt.D, diameter, mvmt.MT) for mvmt in movements]
    times = [mvmt.time for mvmt in movements]
    IPs_raw = [calc_IP(mvmt.D_raw, diameter, mvmt.MT) for mvmt in movements]
    IPs_corr0 = [calc_IP(mvmt.D_corr0, diameter, mvmt.MT) for mvmt in movements]
    aim_strains = [mvmt.aim_strain for mvmt in movements]

    return (miss_probs, IPs, times, IPs_raw, IPs_corr0, aim_strains, tap_strains)


def remove_spinners(beatmap):
    beatmap['hitObjects'] = [obj for obj in beatmap['hitObjects'] if obj['objectName'] != 'spinner']


def apply_mods(beatmap, mods):

    hit_objects = beatmap['hitObjects']
    cs = float(beatmap['CircleSize'])

    if mods[0] == "hr":
        cs = 1.3 * cs
    elif mods[0] == "ez":
        cs = 0.5 * cs

    beatmap["CsAfterMods"] = cs

    if mods[1] == "dt":
        speed_up(hit_objects, 1.5)
    elif mods[1] == "ht":
        speed_up(hit_objects, 0.75)


def speed_up(hit_objects, factor):

    for obj in hit_objects:

        obj["startTime"] = obj["startTime"] / factor

        if "endTime" in obj:
            obj["endTime"] = obj["endTime"] / factor


# def calc_aim_diff_naive(beatmap):

#     hit_objects = beatmap['hitObjects']
#     diameter = cs_to_diameter(beatmap["CsAfterMods"])
#     movements = []

#     for obj1, obj2 in zip(hit_objects, hit_objects[1:]):
#         movements.append(extract_D_MT(diameter, obj1, obj2))

#     TP = calc_throughput(movements, diameter)
    
#     diff = TP / 2.3
#     return diff




def calc_tap_diff(beatmap, analysis=False):
    
    # k = np.array([0.3, 1.5, 7.5])
    # k = np.exp(np.linspace(1.6, -2, num=9))
    k = np.exp(np.linspace(1.7, -1.6, num=4))

    hit_objects = beatmap['hitObjects']
    curr_strain = np.zeros_like(k)
    prev_time = 0.0
    max_strain = np.zeros_like(k)
    strain_history = []

    for obj in hit_objects:
        
        curr_time = obj['startTime'] / 1000
        curr_strain *= np.exp(-k * (curr_time - prev_time))

        if analysis:
            strain_history.append((list(curr_strain), curr_time))

        max_strain = np.maximum(max_strain, curr_strain)
        obj['tapStrain'] = curr_strain.copy()

        curr_strain += k
        prev_time = curr_time

    diff = np.average(max_strain) ** 0.85 * 0.768

    if analysis:
        return strain_history
    else:
        return (diff,) + tuple(max_strain)




def calc_distance(pos1, pos2):
    return ((pos1[0]-pos2[0]) ** 2 + (pos1[1]-pos2[1]) ** 2) ** 0.5


def get_finish_position(slider):
    if slider['repeatCount'] % 2 == 0:
        return slider['position']
    else:
        return slider['endPosition']


def calc_IP_vs_time(file_path):
        
    bm = load_beatmap(file_path)

    hit_objects = bm['hitObjects']
    cs = float(bm['CircleSize'])
    diameter = cs_to_diameter(cs)

    movements = []
    times = []

    for obj1, obj2 in zip(hit_objects, hit_objects[1:]):
        movements.append(extract_D_MT(diameter, obj1, obj2))
        times.append(obj2['startTime'])

    IPs = [calc_IP(D, diameter, MT) for (D, MT) in movements]

    return (IPs, times)


if __name__ == "__main__":
    
    name = sys.argv[1]

    if len(sys.argv) >= 3:
        mods_str = sys.argv[2]
    else:
        mods_str = '-'

    diff = calc_file_diff('data/maps/' + name + '.json', str_to_mods(mods_str))

    msg = ' | '.join(['{:6.3f}'.format(x) for x in diff])

    print(msg)
