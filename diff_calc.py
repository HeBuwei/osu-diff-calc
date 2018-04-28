import json
import sys

import numpy as np

from fitts_law import calc_throughput, calc_IP


def calc_file_diff(file_path, mods=["nm", "nm"]):
    bm = load_beatmap(file_path)
    return calc_diff(bm, mods)


def calc_diff(beatmap, mods=["nm", "nm"]):
    
    remove_spinners(beatmap)
    apply_mods(beatmap, mods)
    
    aim_diff = calc_aim_diff_corrected(beatmap)
    tap_diff = calc_tap_diff(beatmap)

    i = 7.0
    overall_diff = ((aim_diff ** i + tap_diff ** i) / 2) ** (1/i)

    return (aim_diff, tap_diff, overall_diff)


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


def calc_aim_diff_naive(beatmap):

    hit_objects = beatmap['hitObjects']
    diameter = cs_to_diameter(beatmap["CsAfterMods"])
    Ds_MTs = []

    for obj1, obj2 in zip(hit_objects, hit_objects[1:]):
        Ds_MTs.append(extract_D_MT(diameter, obj1, obj2))

    TP = calc_throughput(Ds_MTs, diameter)
    
    diff = TP / 2.3
    return diff


def calc_aim_diff_corrected(beatmap):

    hit_objects = beatmap['hitObjects']
    diameter = cs_to_diameter(beatmap["CsAfterMods"])
    Ds_MTs = []

    for obj1, obj2, obj3 in zip(hit_objects, hit_objects[1:], hit_objects[2:]):
        Ds_MTs.append(extract_D_MT_corrected(diameter, obj1, obj2, obj3=obj3))

    Ds_MTs.append(extract_D_MT_corrected(diameter, hit_objects[-2], hit_objects[-1]))

    TP = calc_throughput(Ds_MTs, diameter)

    diff = TP / 2.4
    return diff


# Extract distance (D) and movement time (MT) between 2 hit objects
def extract_D_MT(diameter, obj1, obj2):
    
    if obj1['objectName'] == 'slider':

        finishPosition = get_finish_position(obj1)

        # long sliders (when the slider tail matters)
        D_long = max(calc_distance(finishPosition, obj2['position']) - 1.5 * diameter, 0.0)
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
        D = calc_distance(obj1['position'], obj2['position'])
        MT = (obj2['startTime'] - obj1['startTime']) / 1000.0
    
    else:
        raise Exception
    
    return (D, MT)


# Extract D and MT of the movement from obj1 and obj2 and adjust the values
# by taking the next object into consideration
def extract_D_MT_corrected(diameter, obj1, obj2, obj3=None):
    
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


    # Estimate how obj3 affects the difficulty of obj1 -> obj2 and make correction accordingly
    # Very empirical, may need tweaking
    if obj3 is not None:

        s1 = np.array(obj2['position']) - np.array(finish_position)
        s2 = np.array(obj3['position']) - np.array(obj2['position'])

        if np.sqrt(s1.dot(s1)) == 0:
            correction_obj3 = 0
        else:
            correction_obj3 = min((s2.dot(s1) / np.sqrt(s1.dot(s1)) + np.sqrt(s2.dot(s2))) / diameter * 0.05, 0.2)
        
        t2 = (obj3['startTime'] - obj2['startTime']) / 1000.0
        correction_obj3 = correction_obj3 * max(1 - 2.5 * t2, 0)

        D = D * (1 + correction_obj3)
    
    return (D, MT)


def calc_tap_diff(beatmap):
    
    hit_objects = beatmap['hitObjects']
    curr_strain = 0.0
    prev_time = 0
    max_strain = 0.0
    # strain_history = []

    for obj in hit_objects:
        curr_time = obj['startTime']
        curr_strain *= 0.25 ** ((curr_time - prev_time) / 1000.0)
        curr_strain += 1.0
        # strain_history.append((curr_strain, curr_time))
        max_strain = max(max_strain, curr_strain)
        prev_time = curr_time

    diff = max_strain / 1.6
    return diff


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

    Ds_MTs = []
    times = []

    for obj1, obj2 in zip(hit_objects, hit_objects[1:]):
        Ds_MTs.append(extract_D_MT(diameter, obj1, obj2))
        times.append(obj2['startTime'])

    IPs = [calc_IP(D, diameter, MT) for (D, MT) in Ds_MTs]

    return (IPs, times)


def load_beatmap(file_path):
    with open(file_path, encoding="utf8") as bm_file:
        bm = json.load(bm_file)
        return bm


if __name__ == "__main__":
    
    name = sys.argv[1]
    mods = [sys.argv[2], sys.argv[3]]

    diff = calc_file_diff('data/maps/' + name + '.json', mods)

    print(diff)
