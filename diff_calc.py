import sys

import numpy as np
from scipy.special import expit
from scipy.linalg import norm

from mods import str_to_mods
from beatmap import load_beatmap
from fitts_law import calc_IP, calc_hit_prob
from aim import calc_aim_diff, cs_to_diameter
from tap import calc_tap_diff


def calc_file_diff(file_path, mods=["nm", "nm"], tap_detail=False):
    beatmap = load_beatmap(file_path)
    return calc_diff(beatmap, mods, tap_detail)


def calc_diff(beatmap, mods=["nm", "nm"], tap_detail=False):
    
    remove_spinners(beatmap)
    apply_mods(beatmap, mods)
    
    tap_diff = calc_tap_diff(beatmap)
    aim_diff = calc_aim_diff(beatmap)

    i = 7.0
    overall_diff = ((aim_diff ** i + tap_diff[0] ** i)/2) ** (1/i) * 1.069

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
    print(" Aim      Tap     Overall")
    print(msg)
