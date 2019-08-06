import sys
import math
from collections import namedtuple

import numpy as np
from scipy.special import expit
from scipy.linalg import norm
from scipy import optimize

from mods import str_to_mods
from beatmap import load_beatmap
from fitts_law import calc_IP, calc_hit_prob


P_THRESHOLD = 0.02

AIM_MIN = 0.1
AIM_MAX = 100

Movement = namedtuple('Movement', ['D', 'MT', 'time', 'D_raw', 'D_corr0', 'aim_strain'])
State = namedtuple('State', ['t', 'p', 'sigma_p', 'v', 'sigma_v'])


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


def calc_aim_diff(beatmap, analysis=False):

    hit_objects = beatmap['hitObjects']
    W = cs_to_diameter(beatmap["CsAfterMods"])

    aim = calc_aim_skill(hit_objects, W)

    diff = aim ** 0.85 * 0.618

    if analysis:
        return (aim, adjusted_movements)
    else:
        return diff


def calc_aim_skill(hit_objects, W):

    fc_prob_aim_min = calc_fc_prob(AIM_MIN, hit_objects, W)
    
    # if map is so easy that players with minimum throughput can fc with decent possibility
    if fc_prob_aim_min >= P_THRESHOLD:
        return AIM_MIN

    fc_prob_aim_max = calc_fc_prob(AIM_MAX, hit_objects, W)

    # if map is too hard 
    if fc_prob_aim_max <= P_THRESHOLD:
        return AIM_MAX

    # x, r = optimize.brentq(calc_fc_prob_minus_threshold, AIM_MIN, AIM_MAX, args=(hit_objects, W), full_output=True)
    # print(r.iterations)

    aim = optimize.brentq(calc_fc_prob_minus_threshold, AIM_MIN, AIM_MAX, args=(hit_objects, W))

    return aim


# def extract_movements(hit_objects, diameter):

#     movements = []

#     if len(hit_objects) == 2:

#         movements.append(extract_movement(diameter, hit_objects[0], hit_objects[1]))

#     elif len(hit_objects) >= 3:

#         movements.append(extract_movement(diameter, hit_objects[0], hit_objects[1], obj3=hit_objects[2]))

#         for obj0, obj1, obj2, obj3 in zip(hit_objects, hit_objects[1:], hit_objects[2:], hit_objects[3:]):
#             movements.append(extract_movement(diameter, obj1, obj2, obj0=obj0, obj3=obj3))

#         movements.append(extract_movement(diameter, hit_objects[-2], hit_objects[-1], obj0=hit_objects[-3]))

#     return movements


# Returns a new list of movements, of which the MT is adjusted based on aim strain
# def calc_adjusted_movements(movements, diameter):

#     adjusted_movements = []
#     curr_strain = 0
#     prev_time = 0.0
#     k = 2

#     for mvmt in movements:

#         curr_time = mvmt.time
#         curr_strain *= np.exp(-k * (curr_time - prev_time))

#         IP_old = calc_IP(mvmt.D, diameter, mvmt.MT)

#         if IP_old == 0:
#             adjustment = 0
#         else:
#             adjustment = expit((curr_strain / IP_old - 0.7) * 15) * 0.1 - 0.05

#         MT_new = mvmt.MT / (1 + adjustment)

#         adjusted_movements.append(mvmt._replace(aim_strain=curr_strain))
#         # adjusted_movements.append(mvmt._replace(MT=MT_new, aim_strain=curr_strain))

#         curr_strain += np.log2(mvmt.D / diameter + 1) * k
#         prev_time = curr_time

#     return adjusted_movements


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

    
    IP = calc_IP(D, diameter, MT)


    # Correction #1 - The Previous Object
    # Estimate how obj0 affects the difficulty of obj1 -> obj2 and make correction accordingly
    # Very empirical, may need tweaking
    correction_snap = 0
    correction_flow = 0

    if obj0 is not None:
        
        s01 = (np.array(obj1['position']) - np.array(obj0['position'])) / diameter
        s12 = (np.array(obj2['position']) - np.array(finish_position)) / diameter
        
        if s12.dot(s12) == 0:          
            correction_lvl_snap = 0
            correction_lvl_flow = 0
            correction_snap = 0

        else:
            t01 = (obj1['startTime'] - obj0['startTime']) / 1000.0
            t12 = MT

            v01 = s01 / t01
            v12 = s12 / t12
            dv = v12 - v01

            a01 = -4 * s01 / t01 ** 2
            a12 = 4 * s12 / t12 ** 2
            da = a12 - a01


            # Version 1
            correction_lvl_flow = dv.dot(dv) / (2 * (v12.dot(v12) + v01.dot(v01)))
            correction_lvl_snap = da.dot(da) / (2 * (a12.dot(a12) + a01.dot(a01)))

            flowiness = expit((norm(s12) - 1.45) * (-10))
            snappiness = expit((norm(s12) - 1.4) * 10)

            correction_flow = 0.5 ** (t01 / t12 / 2) * correction_lvl_flow * flowiness * 1
            correction_snap = 0.5 ** (t01 / t12 / 2) * correction_lvl_snap * snappiness * 1



    # Correction #2 - The Next Object
    # Estimate how obj3 affects the difficulty of obj1 -> obj2 and make correction accordingly
    # Again, very empirical
    correction_obj3 = 0

    # if obj3 is not None:

    #     s1 = (np.array(obj1['position']) - np.array(obj2['position'])) / diameter
    #     s3 = (np.array(obj3['position']) - np.array(obj2['position'])) / diameter

    #     # t1 = -1
    #     # t3 = (obj3['startTime'] - obj2['startTime']) / (obj2['startTime'] - obj1['startTime'])

    #     t1 = (obj1['startTime'] - obj2['startTime']) / 1000
    #     t3 = (obj3['startTime'] - obj2['startTime']) / 1000

    #     x_params = np.linalg.solve(np.array([[t1*t1/2, t1],
    #                                          [t3*t3/2, t3]]),
    #                                np.array([s1[0], s3[0]]))

    #     y_params = np.linalg.solve(np.array([[t1*t1/2, t1],
    #                                          [t3*t3/2, t3]]),
    #                                np.array([s1[1], s3[1]]))

    #     a = np.array([x_params[0], y_params[0]])
    #     v = np.array([x_params[1], y_params[1]])


    #     print(obj2['startTime'], ", ",
    #           ' | '.join(['{:6.3f}'.format(x) for x in [norm(v), norm(v)/(IP+0.1), v[0], v[1], a[0], a[1]]]))
        # print(a, ", ", v)



    # Correction #3 - Tap Strain
    # Estimate how tap strain affects difficulty
    correction_tap = 0

    if 'tapStrain' in obj2 and D > 0:
        tap_strain = obj2['tapStrain']
        correction_tap = expit((np.average(tap_strain) / IP - 1) * 15) * 0.2



    # Correction #4 - Cheesing
    # The player might make the movement of obj1 -> obj2 easier by 
    # hitting obj1 early and obj2 late. 

    correction_early = correction_late = 0

    if D > 0:

        if obj0 is not None:
            D01 = calc_distance(obj0['position'], obj1['position'])
            MT01 = (obj1['startTime'] - obj0['startTime']) / 1000.0
            MT01_recp = 1 / MT01
            IP01 = calc_IP(D01, diameter, MT01)
        else:
            MT01_recp = 0
            IP01 = 0

        correction_early = expit((IP01 / IP - 0.6) * (-15)) * (1 / (1/(MT+0.07) + MT01_recp)) * 0.12

        if obj3 is not None:
            D23 = calc_distance(obj2['position'], obj3['position'])
            MT23 = (obj3['startTime'] - obj2['startTime']) / 1000.0
            MT23_recp = 1 / MT23
            IP23 = calc_IP(D23, diameter, MT23)
        else:
            MT23_recp = 0
            IP23 = 0

        correction_late = expit((IP23/IP - 0.6) * (-15)) * (1 / (1/(MT+0.07) + MT23_recp)) * 0.12


    # apply the corrections

    D_raw = D
    # MT -= 0.05 * correction_snap
    # D *= 1 + correction_snap
    
    D_corr0 = D_raw + correction_flow * D_raw * 1.5 + correction_snap * 1.5 * diameter

    D_corr_tap = D_corr0 * (1 + correction_tap)
    # D_corr_tap = D_corr0

    MT += correction_early + correction_late

    return Movement(D_corr_tap, MT, obj2['startTime'] / 1000, D_raw, D_corr0, 0)



def calc_fc_prob_minus_threshold(aim, hit_objects, W):
    return calc_fc_prob(aim, hit_objects, W) - P_THRESHOLD
    

def calc_fc_prob(aim, hit_objects, W):

    fc_prob = 1.0

    depth = 4

    if len(hit_objects) < depth:
        return fc_prob

    

    for i in range(len(hit_objects) - depth + 1):

        objs = hit_objects[i:i+depth]
        state0 = State(obj0['startTime'], obj0['position'], 0, [0, 0], 0)

        states = [[] for x in range(depth)]
        states[0].append(state0)

        for j in range(0, depth - 1):

            for state in states[j]:
            
                next_state = calc_next_state(aim, state, objs[j+1])





        # fc_prob *= math.erf(2.066 * (2**aim-1) / 2**0.5)

    return fc_prob

    
def calc_next_state(aim, state, obj):

    





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

    msg = ' | '.join(['{:6.3f}'.format(x) for x in diff])

    print(msg)
