import sys
from collections import namedtuple

import numpy as np
from scipy.special import expit
from scipy.linalg import norm

from mods import str_to_mods
from beatmap import load_beatmap
from fitts_law import calc_throughput, calc_IP, calc_hit_prob


Movement = namedtuple('Movement', ['D', 'MT', 'time', 'D_raw', 'D_corr0'])


def calc_file_diff(file_path, mods=["nm", "nm"]):
    beatmap = load_beatmap(file_path)
    return calc_diff(beatmap, mods)


def calc_diff(beatmap, mods=["nm", "nm"]):
    
    remove_spinners(beatmap)
    apply_mods(beatmap, mods)
    
    tap_diff = calc_tap_diff(beatmap)
    aim_diff = calc_aim_diff(beatmap)

    i = 7.0
    overall_diff = (aim_diff ** i + tap_diff[0] ** i) ** (1/i) * 0.968

    return (aim_diff,) + tap_diff[:1] + (overall_diff,)
    # return (aim_diff,) + tap_diff + (overall_diff,)


def analyze_file_diff(file_path, mods=["nm", "nm"]):
    beatmap = load_beatmap(file_path)
    return analyze_diff(beatmap, mods)


def analyze_diff(beatmap, mods=["nm", "nm"]):

    remove_spinners(beatmap)
    apply_mods(beatmap, mods)
    
    cs = float(beatmap['CircleSize'])
    diameter = cs_to_diameter(cs)

    strain_history = calc_tap_diff(beatmap, analysis=True)
    TP, movements = calc_aim_diff(beatmap, analysis=True)

    miss_probs = [1 - calc_hit_prob(mvmt.D, diameter, mvmt.MT, TP) for mvmt in movements]
    IPs = [calc_IP(mvmt.D, diameter, mvmt.MT) for mvmt in movements]
    times = [mvmt.time for mvmt in movements]
    IPs_raw = [calc_IP(mvmt.D_raw, diameter, mvmt.MT) for mvmt in movements]
    IPs_corr0 = [calc_IP(mvmt.D_corr0, diameter, mvmt.MT) for mvmt in movements]

    return (miss_probs, IPs, times, IPs_raw, IPs_corr0, strain_history)


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


def calc_aim_diff(beatmap, analysis=False):

    hit_objects = beatmap['hitObjects']
    diameter = cs_to_diameter(beatmap["CsAfterMods"])
    movements = []

    if len(hit_objects) == 2:

        movements.append(extract_movement(diameter, hit_objects[0], hit_objects[1]))

    elif len(hit_objects) >= 3:

        movements.append(extract_movement(diameter, hit_objects[0], hit_objects[1], obj3=hit_objects[2]))

        for obj0, obj1, obj2, obj3 in zip(hit_objects, hit_objects[1:], hit_objects[2:], hit_objects[3:]):
            movements.append(extract_movement(diameter, obj1, obj2, obj0=obj0, obj3=obj3))

        movements.append(extract_movement(diameter, hit_objects[-2], hit_objects[-1], obj0=hit_objects[-3]))

    TP = calc_throughput(movements, diameter)
    diff = TP ** 0.85 * 0.594

    if analysis:
        return (TP, movements)
    else:
        return diff


# Extract distance (D) and movement time (MT) between 2 hit objects
# def extract_D_MT(diameter, obj1, obj2):
    
#     if obj1['objectName'] == 'slider':

#         finishPosition = get_finish_position(obj1)

#         # long sliders (when the slider tail matters)
#         D_long = max(calc_distance(finishPosition, obj2['position']) - 1.5 * diameter, 0.0)
#         MT_long = (obj2['startTime'] - obj1['endTime'] + 70) / 1000.0

#         # short sliders (when the slider head matters) (treat as a circle)
#         D_short = calc_distance(obj1['position'], obj2['position'])
#         MT_short = (obj2['startTime'] - obj1['startTime']) / 1000.0

#         if calc_IP(D_long, diameter, MT_long) > calc_IP(D_short, diameter, MT_short):
#             D = D_long
#             MT = MT_long
#         else:
#             D = D_short
#             MT = MT_short

#     elif obj1['objectName'] == 'circle':
#         D = calc_distance(obj1['position'], obj2['position'])
#         MT = (obj2['startTime'] - obj1['startTime']) / 1000.0
    
#     else:
#         raise Exception
    
#     return (D, MT)


# Extract information about the movement from obj1 and obj2
# including D, MT, end time of the movement, and the factors of correction
def extract_movement(diameter, obj1, obj2, obj0=None, obj3=None):
    
    if obj1['objectName'] == 'slider':

        finish_position = get_finish_position(obj1)

        # long sliders (when the slider tail matters)
        D_long = max(calc_distance(finish_position, obj2['position']) - 1.5 * diameter, 0.0)
        MT_long = (obj2['startTime'] - obj1['endTime'] + 70) / 1000.0

        # short sliders (when the slider head matters) (treat as a circle)
        D_short = calc_distance(obj1['position'], obj2['position'])
        MT_short = (obj2['startTime'] - obj1['startTime']) / 1000.0

        if calc_IP(D_long, diameter, MT_long) > calc_IP(D_short, diameter, MT_short):
            D = D_long
            MT = MT_long
        else:
            D = D_short
            MT = MT_short

    elif obj1['objectName'] == 'circle':

        finish_position = obj1['position']

        D = calc_distance(obj1['position'], obj2['position'])
        MT = (obj2['startTime'] - obj1['startTime']) / 1000.0
    
    else: 
        raise Exception  


    # Estimate how obj0 affects the difficulty of obj1 -> obj2 and make correction accordingly
    # Very empirical, may need tweaking
    correction_obj0 = 0

    if obj0 is not None:
        
        s01 = (np.array(obj1['position']) - np.array(obj0['position'])) / diameter
        s12 = (np.array(obj2['position']) - np.array(finish_position)) / diameter
        
        t01 = (obj1['startTime'] - obj0['startTime']) / 1000.0
        t12 = MT

        a01 = -4 * s01 / t01 ** 2
        a12 = 4 * s12 / t12 ** 2

        if a12.dot(a12) == 0:
            correction_lvl_obj0 = 0
        else:
            da = a12 - a01
            # correction_lvl_obj0 = (norm(da) / (norm(a12) + norm(a01))) ** 2
            correction_lvl_obj0 = da.dot(da) / (2 * (a12.dot(a12) + a01.dot(a01)))

        # if np.sqrt(s12.dot(s12)) == 0:
        #     correction_lvl_obj0 = 0
        # else:
        #     correction_lvl_obj0 = (s20.dot(-s12) / np.sqrt(s12.dot(s12)) + np.sqrt(s20.dot(s20))) / diameter / 2

        snappiness = expit((norm(s12) - 1.3) * 4)
        correction_obj0 = 0.5 ** (t01 * 5) * correction_lvl_obj0 * snappiness * 1

    


    # Estimate how obj3 affects the difficulty of obj1 -> obj2 and make correction accordingly
    # Again, very empirical
    # correction_obj3 = 0

    # if obj3 is not None:

    #     s1 = (np.array(obj2['position']) - np.array(finish_position)) / diameter 
    #     s2 = (np.array(obj3['position']) - np.array(obj2['position'])) / diameter 

    #     if s1.dot(s1) == 0:
    #         correction_lvl_obj3 = 0
    #     else:
    #         correction_lvl_obj3 = (s2.dot(s1) / np.sqrt(s1.dot(s1)) + np.sqrt(s2.dot(s2))) / 2
        
    #     t2 = (obj3['startTime'] - obj2['startTime']) / 1000.0
    #     correction_obj3 = min(0.5, max(0, correction_lvl_obj3 - 1)) * max(0, 1 - 2.5 * t2)

        # D *= 1 + correction_obj3


    correction_tap = 0

    if 'tapStrain' in obj2 and D > 0:

        tap_strain = obj2['tapStrain']
        IP = calc_IP(D, diameter, MT)

        correction_tap = expit((np.average(tap_strain) / IP - 1) * 12) * 0.2


    # apply the corrections

    D_raw = D
    # MT -= 0.05 * correction_obj0
    # D *= 1 + correction_obj0
    
    D_corr0 = D_raw + correction_obj0 * diameter * 2

    D_corr_tap = D_corr0 * (1 + correction_tap)
    # D_corr_tap = D_corr0

    return Movement(D_corr_tap, MT, obj2['startTime'], D_raw, D_corr0)


def calc_tap_diff(beatmap, analysis=False):
    
    # k = np.array([0.3, 1.5, 7.5])
    # k = np.exp(np.linspace(1.6, -2, num=9))
    k = np.exp(np.linspace(1.7, -1.5, num=4))

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

    diff = np.average(max_strain) ** 0.85 * 0.765

    if analysis:
        return strain_history
    else:
        return (diff,) + tuple(max_strain)


def cs_to_diameter(cs):
    # formula: (32.01*(1-(0.7*(cs-5)/5))) * 2
    return 108.834 - 8.9628 * cs


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

    print(diff)
